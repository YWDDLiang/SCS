import unittest

import torch

from src.data.selection.representation_spectrum_scheduler import RepresentationSpectrumScheduler


class RepresentationSpectrumSchedulerTest(unittest.TestCase):
    def test_budget_is_measured_against_all_auxiliary_slots(self):
        scheduler = RepresentationSpectrumScheduler(projection_dim=4, target_view_rate=0.75, seed=3)
        scores = torch.arange(8, dtype=torch.float32)
        eligible = torch.tensor([True, False, True, False, True, False, True, False])
        chosen = scheduler.choose_views(scores, eligible, budget_size=8)
        self.assertEqual(int(chosen.sum()), 4)
        self.assertTrue(torch.equal(torch.nonzero(chosen).flatten(), torch.tensor([0, 2, 4, 6])))

    def test_streaming_state_round_trip(self):
        scheduler = RepresentationSpectrumScheduler(projection_dim=8, seed=9)
        scheduler.score_and_observe(torch.randn(12, 16))
        state = scheduler.state_dict()
        restored = RepresentationSpectrumScheduler(projection_dim=8, seed=9)
        restored.load_state_dict(state)
        self.assertEqual(restored.observed, 12)
        self.assertTrue(torch.equal(restored.projection, scheduler.projection.cpu()))
        self.assertTrue(torch.equal(restored.covariance, scheduler.covariance.cpu()))

    def test_dimension_change_is_rejected(self):
        scheduler = RepresentationSpectrumScheduler(projection_dim=4, seed=1)
        scheduler.score_and_observe(torch.randn(3, 7))
        with self.assertRaises(ValueError):
            scheduler.score_and_observe(torch.randn(3, 8))

    def test_stratified_budget_preserves_group_counts(self):
        scheduler = RepresentationSpectrumScheduler(projection_dim=4, target_view_rate=0.5, seed=3)
        scores = torch.tensor([9., 8., 7., 6., 5., 4., 3., 2.])
        eligible = torch.ones(8, dtype=torch.bool)
        strata = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        chosen = scheduler.choose_views(scores, eligible, budget_size=8, strata=strata)
        self.assertEqual(int(chosen.sum()), 4)
        self.assertEqual(int(chosen[:4].sum()), 2)
        self.assertEqual(int(chosen[4:].sum()), 2)
        self.assertTrue(torch.equal(torch.nonzero(chosen).flatten(), torch.tensor([1, 3, 5, 7])))


if __name__ == "__main__":
    unittest.main()

