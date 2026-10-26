import unittest

import torch

from src.data.selection.conflict_aware_presentation import ConflictAwarePresentationPolicy


class ConflictAwarePresentationPolicyTest(unittest.TestCase):
    def test_probabilities_are_finite_and_sum_to_one(self):
        policy = ConflictAwarePresentationPolicy(seed=7)
        probabilities = policy.event_probabilities(torch.tensor([1.0, 2.0, 5.0]))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_conflict_bounds_and_order(self):
        policy = ConflictAwarePresentationPolicy(min_augmentation_rate=0.5, max_augmentation_rate=0.9)
        probabilities, target = policy.augmentation_probabilities(torch.tensor([0.0, 0.2, 1.0]))
        self.assertGreaterEqual(float(target), 0.5)
        self.assertLessEqual(float(target), 0.9)
        self.assertGreaterEqual(float(probabilities[2]), float(probabilities[1]))
        self.assertGreaterEqual(float(probabilities[1]), float(probabilities[0]))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_conflict_action_mass_is_calibrated_under_spectrum_distribution(self):
        policy = ConflictAwarePresentationPolicy(min_augmentation_rate=0.5, max_augmentation_rate=0.9)
        conflict = torch.tensor([0.0, 0.1, 0.4, 1.0])
        event_probabilities = torch.tensor([0.05, 0.15, 0.30, 0.50])
        action, target = policy.augmentation_probabilities(conflict, event_probabilities)
        self.assertAlmostEqual(
            float((action * event_probabilities).sum()), float(target), places=5
        )
        self.assertLessEqual(float(action[0]), float(action[1]))
        self.assertLessEqual(float(action[1]), float(action[2]))
        self.assertLessEqual(float(action[2]), float(action[3]))

    def test_all_zero_conflict_uses_bounded_uniform_rate(self):
        policy = ConflictAwarePresentationPolicy(min_augmentation_rate=0.5, max_augmentation_rate=0.9)
        action, target = policy.augmentation_probabilities(
            torch.zeros(3), torch.tensor([0.05, 0.15, 0.80])
        )
        self.assertEqual(float(target), 0.5)
        self.assertTrue(torch.allclose(action, torch.full((3,), 0.5)))

    def test_conflict_rate_maps_linearly_to_adaptive_bounds(self):
        policy = ConflictAwarePresentationPolicy(min_augmentation_rate=0.5, max_augmentation_rate=0.9)
        _, target = policy.augmentation_probabilities(torch.tensor([0.0, 1.0]))
        self.assertAlmostEqual(float(target), 0.7, places=6)

    def test_sampling_is_reproducible(self):
        policy = ConflictAwarePresentationPolicy(seed=11)
        spectrum = torch.tensor([0.2, 0.5, 2.0, 1.0])
        conflict = torch.tensor([0.0, 0.1, 0.8, 0.3])
        g1 = torch.Generator().manual_seed(31)
        g2 = torch.Generator().manual_seed(31)
        a = policy.sample(spectrum, conflict, 100, generator=g1)
        b = policy.sample(spectrum, conflict, 100, generator=g2)
        self.assertTrue(torch.equal(a.event_indices, b.event_indices))
        self.assertTrue(torch.equal(a.augment, b.augment))

    def test_shape_mismatch_is_rejected(self):
        policy = ConflictAwarePresentationPolicy()
        with self.assertRaises(ValueError):
            policy.sample(torch.ones(3), torch.ones(2), 1)


if __name__ == "__main__":
    unittest.main()
