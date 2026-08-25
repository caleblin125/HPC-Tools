import unittest

from parameter_search import EliteRandomSearch, Parameter


class EliteRandomSearchTests(unittest.TestCase):
    def setUp(self):
        self.search = EliteRandomSearch(
            [Parameter("size", [1, 2, 3]), Parameter("mode", ["a", "b"])],
            elite_count=1,
            mutation_count=1,
            seed=3,
        )

    def test_empty_history_is_random(self):
        candidates = self.search.propose([], 3)
        self.assertEqual(3, len(candidates))
        self.assertTrue(all(candidate.origin == "random" for candidate in candidates))

    def test_best_valid_record_is_mutated_first(self):
        candidates = self.search.propose(
            [{"valid": True, "score": 2.0, "size": 1, "mode": "a"},
             {"valid": True, "score": 9.0, "size": 3, "mode": "b"},
             {"valid": False, "score": 99.0, "size": 1, "mode": "a"}],
            2,
        )
        self.assertEqual("mutated-elite", candidates[0].origin)
        self.assertEqual("random", candidates[1].origin)
        self.assertIn(candidates[0].values["size"], [2, 3])
        self.assertEqual({"size", "mode"}, set(candidates[0].values))


if __name__ == "__main__":
    unittest.main()
