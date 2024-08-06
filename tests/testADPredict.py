import unittest
from WIMOAD.ADPredict import ADPredict

class TestADPredict(unittest.TestCase):
    def test_ADPredict(self):
        results = ADPredict()
        self.assertIsNotNone(results)

if __name__ == "__main__":
    unittest.main()
