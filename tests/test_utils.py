"""Le réessai exponentiel : c'est lui qui absorbe les 429 de l'API."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rag.utils import retry


class TestRetry(unittest.TestCase):
    def test_succes_immediat(self) -> None:
        self.assertEqual(retry(lambda: 42), 42)

    def test_reussite_apres_echecs(self) -> None:
        essais = {"n": 0}

        def instable() -> str:
            essais["n"] += 1
            if essais["n"] < 3:
                raise RuntimeError("429 Rate limit exceeded")
            return "ok"

        with patch("rag.utils.time.sleep") as dormir:
            self.assertEqual(retry(instable), "ok")
        self.assertEqual(essais["n"], 3)
        self.assertEqual([appel.args[0] for appel in dormir.call_args_list], [1.0, 2.0])

    def test_delais_exponentiels_plafonnes(self) -> None:
        with patch("rag.utils.time.sleep") as dormir:
            with self.assertRaises(RuntimeError):
                retry(lambda: 1 / 0, attempts=7, max_delay=8.0)
        self.assertEqual(
            [appel.args[0] for appel in dormir.call_args_list], [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
        )

    def test_erreur_propagee_avec_sa_cause(self) -> None:
        with patch("rag.utils.time.sleep"):
            with self.assertRaises(RuntimeError) as capture:
                retry(lambda: 1 / 0, attempts=2, description="l'appel de test")
        self.assertIn("l'appel de test", str(capture.exception))
        self.assertIsInstance(capture.exception.__cause__, ZeroDivisionError)


if __name__ == "__main__":
    unittest.main()
