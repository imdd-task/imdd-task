"""
Test IM/DD dataset
"""
import unittest
import torch
from IMDD import IMDDDataset, IMDDParams


class TestIMMDDDataset(unittest.TestCase):
    """ Test IM/DD dataset """

    def test_dataset(self):
        params = IMDDParams()
        dataset = IMDDDataset(params)

        batch_size = 10000
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True)

        data, targets = next(iter(dataloader))

        self.assertEqual(len(dataset), params.N)
        self.assertEqual(len(dataloader), int(params.N / batch_size))
        self.assertEqual(list(data.shape), [batch_size, params.n_taps])
        self.assertEqual(targets.shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()