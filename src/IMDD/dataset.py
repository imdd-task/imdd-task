""" IM/DD Dataset """
from typing import Any, Union, Tuple, List, Optional
import torch
from IMDD.helpers import get_graylabel
from IMDD.model import IMDDModel, IMDDParams, LCDParams, SSMFParams
from torch.utils.data import Dataset


class IMDDDataset(Dataset):
    """ Dataset for demapping an IM/DD link with PAM-4 """

    def __init__(self, params: IMDDParams, bit_level: bool = False,
                 continuous_sampling: bool = True, train: bool = True,
                 seed: Optional[int] = None) -> None:
        """
        :param params: The IM/DD model parameter set.
        :param bit_level: Bool indicating whether the targets are on
            symbol-level, or on bit-level.
        :param continuous_sampling: If True, `params.N` new samples are
            generated after all samples in the current sequence have been
            accessed. If False, one sequence is generated and reused. Defaults
            to True.
        :param train: Selects the seed used for generating the training resp.
            the testing data. If seed is not None, this becomes ineffective.
            Defaults to True.
        :param seed: Select a manual seed used to generate data. This makes
            `train` ineffective. This can be used to create a validation set.
        """
        if seed is None:
            seed = 12345 if train else 54321

        self.simulator = IMDDModel(params, seed=seed)
        self.bit_level = bit_level
        self._continuous_sampling = continuous_sampling

        self._size = params.N
        self._targets = None
        self._impaired = None
        self._used_indices = torch.zeros(self._size, dtype=bool)
        self.labels = get_graylabel(
            int(torch.log2(torch.tensor(params.alphabet.shape[0]))))

    def set_n_taps(self, n_taps: int):
        """
        Sets the param `n_taps` used in the deployed IMDD model. This changes
        the shape of the returned tensor to (n_taps,).

        :param n_taps: The number of taps to return.
        """
        self.simulator.params.n_taps = n_taps
        self._used_indices = torch.zeros(self._size, dtype=bool)

    def set_noise_power_db(self, noise_power_db: float):
        """
        Sets the param `noise_power_db` used in the deployed IMDD model. 

        :param noise_power_db: The noise power `\sigma^2` in dB.
        """
        self.simulator.params.noise_power_db = noise_power_db
        self._used_indices = torch.zeros(self._size, dtype=bool)

    def _create_sequence(self) -> None:
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
            if self._continuous_sampling:
                self._create_sequence()
            self._used_indices = torch.ones(self._size, dtype=bool)
        assert not (False in self._used_indices[idx])
        self._used_indices[idx] = False

        return self._impaired[idx], self._targets[idx]


class LCDDataset(IMDDDataset):
    """ Dataset for LCD-Task """
    def __init__(self, bit_level: bool = False,
                 continuous_sampling: bool = True, train: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(
            params=LCDParams(), bit_level=bit_level,
            continuous_sampling=continuous_sampling, seed=seed)

class SSMFDataset(IMDDDataset):
    """ Dataset for SSMF-Task """
    def __init__(self, bit_level: bool = False,
                 continuous_sampling: bool = True, train: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(params=SSMFParams(), bit_level=bit_level, continuous_sampling=continuous_sampling, seed=seed)
