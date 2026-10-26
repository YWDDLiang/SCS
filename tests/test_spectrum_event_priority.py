import unittest
from types import SimpleNamespace

import numpy as np

from src.data.selection.spectrum_event_priority import (
    augmentation_eligibility,
    cross_user_transition_coverage,
    coverage_aware_spectrum_probabilities,
    coverage_for_events,
    partition_integer_budget,
    sample_scheduled_presentations,
)


class SpectrumEventPriorityTest(unittest.TestCase):
    def test_eligibility_uses_uncapped_history_length(self):
        rows = np.asarray(
            [
                [7, 0, 1],   # history_length=1
                [7, 0, 2],   # history_length=2, target is not greater
                [7, 1, 3],   # history_length=2 and target is greater
                [7, 1, 2],   # history_length=1
                [7, 10, 40], # long history; visible length is capped later
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(
            augmentation_eligibility(rows),
            np.asarray([False, False, True, False, True]),
        )

    def test_transition_coverage_counts_distinct_users_using_complete_sid_vectors(self):
        store = SimpleNamespace(
            sid_hierarchy=2,
            tokens=np.asarray([1, 10, 2, 20, 3, 30, 1, 10, 2, 20, 1, 99, 2, 20], dtype=np.int32),
            offsets=np.asarray([0, 6, 10, 14], dtype=np.int64),
            user_ids=np.asarray([10, 11, 12], dtype=np.int64),
        )
        coverage = cross_user_transition_coverage(store)
        self.assertEqual(int(coverage[0, 1]), 2)
        self.assertEqual(int(coverage[1, 1]), 2)
        self.assertEqual(int(coverage[2, 1]), 1)
        self.assertEqual(int(coverage[0, 2]), 1)

    def test_coverage_lookup_maps_raw_user_ids_to_candidate_rows(self):
        store = SimpleNamespace(
            sid_hierarchy=1,
            tokens=np.asarray([1, 2, 1, 2], dtype=np.int32),
            offsets=np.asarray([0, 2, 4], dtype=np.int64),
            user_ids=np.asarray([71, 19], dtype=np.int64),
        )
        table = cross_user_transition_coverage(store)
        result = coverage_for_events(
            np.asarray([[19, 0, 1], [71, 0, 1]], dtype=np.int64), store, table
        )
        np.testing.assert_array_equal(result, [2, 2])

    def test_coverage_is_combined_with_the_spectrum_distribution(self):
        probabilities = coverage_aware_spectrum_probabilities(
            np.asarray([0.2, 0.3, 0.5]),
            np.asarray([1, 2, 3]),
            np.asarray([False, True, True]),
            minimum_coverage=2,
            view_readiness_weight=4.0,
        )
        np.testing.assert_allclose(probabilities, [0.0, 0.375, 0.625])

    def test_scheduled_presentations_observe_exact_action_quota(self):
        selected, augment = sample_scheduled_presentations(
            probabilities=np.asarray([0.2, 0.3, 0.5]),
            augmentation_probabilities=np.asarray([0.0, 0.5, 1.0]),
            view_eligible=np.asarray([False, True, True]),
            extra_count=40,
            augmentation_count=17,
            rng=np.random.default_rng(2),
        )
        self.assertEqual(len(selected), 40)
        self.assertEqual(int(augment.sum()), 17)
        self.assertTrue(np.all(np.asarray([False, True, True])[selected[augment]]))
        self.assertGreater(int((selected[~augment] == 0).sum()), 0)

    def test_candidate_pool_is_shared_by_repeats_and_views(self):
        pool = np.asarray([False, True, True])
        selected, augment = sample_scheduled_presentations(
            probabilities=np.asarray([0.0, 0.3, 0.7]),
            augmentation_probabilities=np.asarray([0.0, 1.0, 1.0]),
            view_eligible=np.asarray([False, True, True]),
            candidate_mask=pool,
            extra_count=80,
            augmentation_count=30,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(len(selected), 80)
        self.assertTrue(np.all(pool[selected]))
        self.assertTrue(np.all(np.asarray([False, True, True])[selected[augment]]))

    def test_rank_budget_partition_preserves_global_cap(self):
        budgets = partition_integer_budget(5_038_053, 4)
        self.assertEqual(int(budgets.sum()), 5_038_053)
        self.assertLessEqual(int(budgets.max() - budgets.min()), 1)
        self.assertEqual(budgets.tolist(), [1_259_514, 1_259_513, 1_259_513, 1_259_513])

if __name__ == "__main__":
    unittest.main()
