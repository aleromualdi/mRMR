import unittest
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from mrmr import MRMR


class TestMRMR(unittest.TestCase):

    def setUp(self):

        # generate 2-classes data
        na = nb = 50
        X_a = np.zeros((na, 10))
        X_b = np.zeros((nb, 10))

        # informed feature vectors
        X_a[:, 0] = np.random.normal(3, 1, na)
        X_a[:, 1] = np.random.normal(5, 1, na)
        X_a[:, 2] = X_a[:, 0] + X_a[:, 1]

        # random norm feature vectors
        for i in range(3, 10):
            X_a[:, i] = np.random.normal(0, 1, na)

        for i in range(10):
            X_b[:, i] = np.random.normal(0, 1, nb)

        X = np.concatenate([X_a, X_b], axis=0)

        # discretize data
        discretizer = KBinsDiscretizer(
            n_bins=3, encode='ordinal', strategy='uniform')

        self.X = discretizer.fit_transform(X)
        self.y = np.array([1] * 50 + [0] * 50)

    # @unittest.skip("tem off")
    def test_raise(self):

        mrmr = MRMR(n_features=2)

        # check if ValueError raised
        with self.assertRaises(ValueError):
            mrmr.fit([1,2,3], self.y)

        with self.assertRaises(ValueError):
            mrmr.fit(self.X, y=None)

        with self.assertRaises(ValueError):
            mrmr.fit(self.X, self.y, 1.2)

    # @unittest.skip("tem off")
    def test_mutual_information_target(self):

        mi_vec = MRMR._mutual_information_target(self.X, self.y)

        indices = [v[0] for v in mi_vec]
        vals = [v[1] for v in mi_vec]

        self.assertEqual(set([0, 1, 2]), set(indices[:3]))
        self.assertTrue([vals[0] > vals[i] for i in range(3, len(vals))])
        self.assertTrue([vals[1] > vals[i] for i in range(3, len(vals))])
        self.assertTrue([vals[2] > vals[i] for i in range(3, len(vals))])

    # @unittest.skip("tem off")
    def test_fit(self):

        mrmr = MRMR(n_features=3, k_max=7)
        mrmr.fit(self.X, self.y, threshold=0.1)
        selected_indices = mrmr.fit(self.X, self.y, threshold=0.01)


if __name__ == '__main__':
    unittest.main()
