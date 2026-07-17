"""DiaFoot.AI v2 — Agreement Tests (Phase 4, Commit 21)."""

import numpy as np
import pytest

from src.evaluation.annotator_agreement import (
    compute_majority_vote,
    compute_pairwise_dice,
    fleiss_kappa,
    staple_consensus,
)


class TestPairwiseDice:
    def test_identical_masks(self) -> None:
        mask = np.ones((32, 32), dtype=np.uint8)
        matrix = compute_pairwise_dice([mask, mask, mask])
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, 1.0)

    def test_different_masks(self) -> None:
        m1 = np.zeros((32, 32), dtype=np.uint8)
        m1[:16, :] = 1
        m2 = np.zeros((32, 32), dtype=np.uint8)
        m2[16:, :] = 1
        matrix = compute_pairwise_dice([m1, m2])
        assert matrix[0, 1] < 0.01

    def test_both_empty_is_perfect_agreement(self) -> None:
        # Two annotators both marking "no lesion" agree perfectly (Dice 1.0),
        # consistent with metrics.dice_score / iou_score on empty masks — not 0.0.
        empty = np.zeros((32, 32), dtype=np.uint8)
        matrix = compute_pairwise_dice([empty, empty])
        assert matrix[0, 1] == 1.0


class TestMajorityVote:
    def test_unanimous(self) -> None:
        mask = np.ones((32, 32), dtype=np.uint8)
        result = compute_majority_vote([mask, mask, mask])
        assert result.sum() == 32 * 32

    def test_majority(self) -> None:
        m1 = np.ones((32, 32), dtype=np.uint8)
        m2 = np.ones((32, 32), dtype=np.uint8)
        m3 = np.zeros((32, 32), dtype=np.uint8)
        result = compute_majority_vote([m1, m2, m3])
        assert result.sum() == 32 * 32  # 2/3 agree

    def test_even_split_tie_is_not_foreground(self) -> None:
        # A 50/50 split between two annotators is not a majority and must not
        # be resolved as foreground (>= 0.5 wrongly counted ties as agreement).
        m1 = np.ones((32, 32), dtype=np.uint8)
        m2 = np.zeros((32, 32), dtype=np.uint8)
        result = compute_majority_vote([m1, m2])
        assert result.sum() == 0


class TestStapleConsensus:
    def test_tie_resolves_to_background(self) -> None:
        # Two fully-disagreeing annotators make STAPLE output probability 0.5
        # everywhere. The binarization must resolve that tie to background
        # (> 0.5), consistent with compute_majority_vote — not foreground.
        pytest.importorskip("SimpleITK")
        ones = np.ones((8, 8), dtype=np.uint8)
        zeros = np.zeros((8, 8), dtype=np.uint8)
        result = staple_consensus([ones, zeros])
        assert result.sum() == 0


class TestFleissKappa:
    def test_perfect_agreement(self) -> None:
        # All raters agree on same category
        ratings = np.array([[3, 0], [0, 3], [3, 0]])
        kappa = fleiss_kappa(ratings)
        assert kappa > 0.9
