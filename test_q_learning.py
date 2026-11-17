import unittest
from unittest.mock import patch

import numpy as np

from q_learning import q_learning, make_costs, _step
from baseline import step


class QLearningTests(unittest.TestCase):
    def test_transition_matches_baseline(self):
        costs = make_costs(2., 4)
        for s in range(9):
            for a in range(min(4, 8 - s) + 1):
                for d in range(5):
                    self.assertEqual(_step(s, 6, a, d, costs), step(s, 6, a, d, costs))

    def test_seed_feasibility_and_schedule(self):
        costs = make_costs(2., 4)
        d = np.array([1, 2, 0, 3, 1, 4, 0] * 10)
        w = np.arange(len(d)) % 7
        policy, detail = q_learning(d, w, costs, passes=5, return_details=True)
        np.testing.assert_array_equal(policy, q_learning(d, w, costs, passes=5))
        self.assertEqual(policy.shape, (9, 7))
        self.assertTrue(np.all(policy <= np.minimum(4, 8-np.arange(9))[:, None]))
        self.assertEqual(detail['visits'].sum(), len(d) * 5)
        self.assertAlmostEqual(detail['history'][-1]['epsilon'], .05)

    def test_exact_reward_update(self):
        # Zero demand: ordering one unit costs c + K + h = 2.62 at p=1.
        rng = np.random.default_rng(0)
        rng.random()
        action = int(rng.integers(2))
        self.assertEqual(action, 1)
        _, detail = q_learning([0], [0], make_costs(1., 1), passes=1,
                               alpha=.1, seed=0, return_details=True)
        self.assertAlmostEqual(detail['q'][0, 0, 1], -.262)

    def test_retained_weekday_and_final_successor(self):
        seen = []
        def transition(s, w, a, d, costs):
            seen.append(w)
            return _step(s, w, a, d, costs)
        with patch('q_learning._transition', return_value=transition):
            q_learning([1, 1], [0, 3], make_costs(1., 1), passes=2)
        self.assertEqual(seen, [0, 3, 0, 3])

    def test_invalid_and_zero_cap(self):
        with self.assertRaises(ValueError):
            q_learning([1], [7], make_costs(1., 1))
        with self.assertRaises(ValueError):
            q_learning([], [], make_costs(1., 1))
        np.testing.assert_array_equal(q_learning([0], [0], make_costs(1., 0)), np.zeros((1, 7)))


if __name__ == '__main__':
    unittest.main()
