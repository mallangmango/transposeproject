import unittest
import numpy as np
import pandas as pd
import torch
from mytranspose import mytranspose

class TestTranspose(unittest.TestCase):
    def test_matrix(self):
        x = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(mytranspose(x), expected)

    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        self.assertTrue(mytranspose(df).equals(df.transpose()))

    def test_tensor(self):
        x = torch.tensor([[1, 2], [3, 4]])
        expected = torch.tensor([[1, 3], [2, 4]])
        self.assertTrue(torch.equal(mytranspose(x), expected))

if __name__ == '__main__':
    unittest.main()
    
    def test_dataframe_with_nan_and_str(self):
        D = np.array([1, 2, 3, 4])
        E = np.array(["red", "white", "red", np.nan])
        F = np.array([True, True, True, False])
        df = pd.DataFrame({"d": D, "e": E, "f": F})
        expected = df.transpose()
        self.assertTrue(mytranspose(df).equals(expected))
