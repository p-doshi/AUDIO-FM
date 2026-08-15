"""Stage 1(b): does training-distribution *breadth* explain the 9-model
RSA matrix better than the domain+paradigm framing used through Stage 0?

Per CLAUDE.md's Stage 1(b) gate: "if it doesn't explain better, keep
domain+paradigm as the working explanation rather than forcing the
reframe into the write-up." This script computes both explanations'
fit to the same RSA matrix and reports whichever wins, honestly.

Method: for a given partition of the 9 models into groups, compute
  mean_within  = average RSA over all pairs sharing a group
  mean_between = average RSA over all pairs in different groups
  gap          = mean_within - mean_between
A larger gap means the partition explains more of the RSA matrix's
structure (same-group models agree with each other more than they
agree with other-group models). Compared against a null distribution
from randomly reshuffling model->group assignments (holding group
sizes fixed) to give the gap a rough significance read, given how few
models (9) and pairs (36) there are to work with.

Usage:
    python -m audio_comp.pipelines.breadth_hypothesis_check \
        --rsa-matrix results/rsa_matrix.csv
"""
from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Domain+paradigm framing: the grouping implicit in this project's model
# table / ModelInfo.paradigm fields through Stage 0 -- essentially every
# model in its own paradigm+domain bucket except the two same-domain,
# same-paradigm pairs.
#
# Extended 2026-08-13 for the 7 Tier 2 models added that day. wavlm/
# unispeech_sat/sew/wav2vec2_conformer/mms all share wav2vec2/hubert's
# core masked-prediction objective (unispeech_sat adds an auxiliary
# speaker-aware loss, sew is an efficiency-architecture variant,
# wav2vec2_conformer swaps the encoder block, mms is the same
# architecture/objective at far larger training scale -- none of these
# changes the fundamental paradigm bucket, so all five join
# masked_modeling_speech). data2vec_audio is data2vec-family
# (self-distillation, EMA teacher) like music2vec, but speech domain
# instead of music -- gets its own "data2vec_speech" bucket, the
# direct domain-paradigm-matched sibling of music2vec's "data2vec_music".
# whisper is genuinely distinct from every existing bucket: encoder-
# decoder trained with ASR (transcript) supervision, not masked-modeling,
# not contrastive, not data2vec-style self-distillation -- own bucket.
DOMAIN_PARADIGM_GROUPS = {
    "masked_modeling_speech": ["hubert", "wav2vec2", "wavlm", "unispeech_sat", "sew", "wav2vec2_conformer", "mms"],
    "masked_modeling_music": ["mert", "musicfm"],
    "data2vec_music": ["music2vec"],
    "data2vec_speech": ["data2vec_audio"],
    "contrastive_general": ["clap"],
    "jepa_general": ["audio_jepa"],
    "reconstruction_bioacoustic": ["bird_mae"],
    "supervised_cnn_general": ["panns_cnn14"],
    "supervised_transformer_general": ["ast"],
    "reconstruction_general": ["audiomae"],
    "supervised_cnn_bioacoustic": ["birdnet"],
    "supervised_asr_speech": ["whisper"],
}

# Training-distribution-breadth framing, per CLAUDE.md Stage 1(b)'s
# original sketch, extended to the 4 models added 2026-08-10: panns_cnn14
# and ast both train on AudioSet (527 diverse everyday-sound classes
# spanning speech, music, animals, vehicles, alarms -- genuinely broad
# content), so both join audio_jepa/clap in "broad_mixed". audiomae ALSO
# trains on AudioSet (same breadth-of-DATA argument) -- included here on
# the same principle, categorized by documented training-data breadth,
# not by how well it happens to fit the RSA matrix; if it turns out to be
# a poor fit for this group empirically (its RSA correlations with the
# rest of broad_mixed are in fact much weaker than the other 4 members'
# mutual correlations, ~0.3-0.5 vs ~0.55-0.84), that is itself honest
# information about the limits of a pure data-breadth account, not a
# reason to move it to a category invented to fit better. bird_mae trains
# exclusively on BirdSet (bird vocalizations only) -- as narrow as
# hubert/wav2vec2's speech-only or mert/musicfm's music-only corpora,
# just a third domain, hence "narrow_bioacoustic" rather than folding it
# into an existing one. birdnet (added 2026-08-10) trains exclusively on
# Macaulay Library/Xeno-canto bird recordings -- same narrow bioacoustic
# domain as bird_mae, despite the two having opposite training
# objectives (birdnet: discriminative/supervised; bird_mae: reconstruction)
# -- breadth groups by DATA domain only, not objective, so both belong
# in narrow_bioacoustic together; if their RSA correlation with each
# other is notably weaker than same-domain same-objective pairs
# elsewhere (e.g. hubert-wav2vec2), that would itself be evidence that
# objective matters *within* a fixed domain, worth checking explicitly
# once this result is in, not assumed either way beforehand.
# Extended 2026-08-13 for the 7 Tier 2 models. wavlm/data2vec_audio/
# unispeech_sat/sew/wav2vec2_conformer all train on a single narrow
# speech corpus (960h LibriSpeech, same as plain wav2vec2/hubert) --
# breadth groups by DATA distribution, not paradigm, so all five join
# narrow_speech regardless of their differing objectives/architectures.
#
# mms and whisper do NOT join narrow_speech, despite being speech-only
# in domain -- CLAUDE.md already flags MMS as "the most extreme
# training-distribution-breadth data point in the roster by a wide
# margin" (500k hours, 1400+ languages); putting it in narrow_speech
# would directly contradict that. Whisper is similarly broad within
# speech (680k hours, ~100 languages, diverse web-scraped conditions,
# multiple tasks). Neither is domain-broad the way broad_mixed's
# members are (speech+music+environmental all at once) -- they're
# domain-narrow (speech only) but distribution-broad *within* that
# domain, a genuinely different axis existing_groups didn't have a
# bucket for. New "broad_speech" group, not folded into broad_mixed
# (would conflate domain breadth with within-domain breadth) or
# narrow_speech (would misrepresent the one property CLAUDE.md already
# singled MMS out for). This also gives the breadth hypothesis a real
# boundary-condition test it didn't have before: does broad_speech
# behave more like narrow_speech (domain match) or broad_mixed
# (breadth match)?
BREADTH_GROUPS = {
    "narrow_speech": ["hubert", "wav2vec2", "wavlm", "data2vec_audio", "unispeech_sat", "sew", "wav2vec2_conformer"],
    "narrow_music": ["mert", "musicfm"],
    "narrow_bioacoustic": ["bird_mae", "birdnet"],
    "broad_mixed": ["audio_jepa", "clap", "panns_cnn14", "ast", "audiomae"],
    "broad_speech": ["mms", "whisper"],
    "self_distillation_narrow_music": ["music2vec"],
}


def load_rsa_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)[1:]
        rows = []
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    return header, np.array(rows)


def group_assignment(groups: dict[str, list[str]], models: list[str]) -> list[str]:
    model_to_group = {}
    for group, members in groups.items():
        for m in members:
            model_to_group[m] = group
    missing = [m for m in models if m not in model_to_group]
    if missing:
        raise ValueError(f"models missing a group assignment: {missing}")
    return [model_to_group[m] for m in models]


def within_between_gap(rsa: np.ndarray, assignment: list[str]) -> tuple[float, float, float, int, int]:
    n = len(assignment)
    within_vals, between_vals = [], []
    for i, j in itertools.combinations(range(n), 2):
        val = rsa[i, j]
        if assignment[i] == assignment[j]:
            within_vals.append(val)
        else:
            between_vals.append(val)
    mean_within = float(np.mean(within_vals)) if within_vals else float("nan")
    mean_between = float(np.mean(between_vals)) if between_vals else float("nan")
    return mean_within, mean_between, mean_within - mean_between, len(within_vals), len(between_vals)


def permutation_test(rsa: np.ndarray, assignment: list[str], n_perm: int = 20000, seed: int = 0) -> float:
    """P(random reshuffle of the SAME group sizes achieves a gap >= the
    observed gap). Reshuffles which model gets which group label, keeping
    the group-size distribution fixed, so this tests whether the specific
    model->group assignment matters, not just the number of groups."""
    rng = random.Random(seed)
    n = len(assignment)
    _, _, observed_gap, *_ = within_between_gap(rsa, assignment)
    count_ge = 0
    labels = list(assignment)
    for _ in range(n_perm):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        _, _, gap, *_ = within_between_gap(rsa, shuffled)
        if gap >= observed_gap:
            count_ge += 1
    return count_ge / n_perm


def main(rsa_matrix_path: str) -> None:
    models, rsa = load_rsa_matrix(Path(rsa_matrix_path))

    print(f"Loaded {len(models)}-model RSA matrix: {models}\n")

    for name, groups in [("domain+paradigm", DOMAIN_PARADIGM_GROUPS), ("breadth", BREADTH_GROUPS)]:
        assignment = group_assignment(groups, models)
        mean_within, mean_between, gap, n_within, n_between = within_between_gap(rsa, assignment)
        p = permutation_test(rsa, assignment)
        print(f"=== {name} ===")
        print(f"  groups: {groups}")
        print(f"  mean_within={mean_within:.4f} (n={n_within} pairs), mean_between={mean_between:.4f} (n={n_between} pairs)")
        print(f"  gap = {gap:.4f}")
        print(f"  permutation p-value (gap >= observed by chance) = {p:.4f}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rsa-matrix", default=str(REPO_ROOT / "results" / "rsa_matrix.csv"))
    args = parser.parse_args()
    main(args.rsa_matrix)
