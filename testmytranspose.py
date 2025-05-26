import unittest
import numpy as np
import pandas as pd
import torch
from mytranspose import mytranspose

class TestTranspose(unittest.TestCase):

    def test_dataframe_with_nan_and_str(self):
        # (3) dataframe의 경우
        D = np.array([1, 2, 3, 4])
        E = np.array(["red", "white", "red", np.nan])  # 문자열과 NaN 포함
        F = np.array([True, True, True, False])
        df = pd.DataFrame({"d": D, "e": E, "f": F})

        transposed_df = mytranspose(df)
        expected = df.transpose()

        self.assertTrue(transposed_df.equals(expected))

    def test_tensor_from_numpy(self):
        # (4) pytorch tensor의 경우
        np_array = np.array([[1, 2], [3, 4]])
        tensor_pt = torch.tensor(np_array)
        expected = torch.tensor([[1, 3], [2, 4]])

        transposed_tensor = mytranspose(tensor_pt)

        self.assertTrue(torch.equal(transposed_tensor, expected))

if __name__ == '__main__':
    unittest.main()
