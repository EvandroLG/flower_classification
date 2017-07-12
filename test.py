import unittest
from flowers_classification import FlowerClassification


class TestFlowerClassification(unittest.TestCase):
    def test_distance_between_points(self):
        p1 = [5.9,  3. ,  5.1,  1.8]
        p2 = [5.9,  3. ,  5.1,  1.8]

        f = FlowerClassification()
        output = f._distance_between_points([5.9,  3. ,  5.1,  1.8],
                [5.9,  3. ,  5.1,  1.8])

        self.assertEqual(0.0, output)

    def test_get_classification(self):
        f = FlowerClassification()
        result = f.get_classification([6.,  3. ,  5.1,  1.8])

        self.assertEqual('virginica', result)


if __name__ == '__main__':
    unittest.main()
