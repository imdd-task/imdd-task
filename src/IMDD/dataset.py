""" IM/DD Dataset """
from typing import Any, Union, Tuple, List
import torch
import numpy as np
from .helpers import get_graylabel
from .channel_model import IMDDModel, IMDDParams
from torch.utils.data import Dataset


class PAM4IMDD(Dataset):
    """ Dataset for demapping an IM/DD link with PAM-4 """

    def __init__(self, params: IMDDParams, bit_level: bool = False) -> None:
        """
        :param params: The IM/DD model parameter set.
        :param bit_level: Bool indicating whether the targets are on
            symbol-level, or on bit-level.
        """
        self.simulator = IMDDModel(params)
        self.bit_level = bit_level

        self._size = params.N
        self._targets = None
        self._impaired = None
        self._used_indices = torch.zeros(self._size, dtype=bool)
        self.labels = torch.tensor(
            get_graylabel(int(np.log2(len(params.alphabet)))))

    def _create_sequence(self) -> None:
        """
        Create a new sequence of targets and send them through the channel
        """
        messages = self.simulator.source()
        self._targets = self.labels[messages, :].float() if self.bit_level \
            else messages
        self._impaired = self.simulator(messages)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: Union[List, Any]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param idx: Index at for which the data is returned from the dataset.
        :returns: Tuple of the (impaired) sample returned by the receiver and
            the actual send target.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Create new data if all data is used
        if isinstance(idx, list):
            assert len(idx) == 1
        if True not in self._used_indices:
            self._create_sequence()
            self._used_indices = torch.ones(self._size, dtype=bool)
        assert not (False in self._used_indices[idx])
        self._used_indices[idx] = False

        return self._impaired[idx], self._targets[idx]