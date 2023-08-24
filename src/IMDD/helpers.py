""" Helper functions """
from typing import Union
import torch
import numpy as np


def get_graylabel(number_bits: int) -> np.ndarray:
    """ """
    if number_bits == 1:
        return np.array([[0], [1]], dtype='uint8')
    label = get_graylabel(number_bits - 1)
    lower_half = np.hstack(
        (np.zeros((2**(number_bits - 1), 1), dtype='uint8'), label))
    upper_half = np.hstack(
        (np.ones((2**(number_bits - 1), 1), dtype='uint8'), np.flipud(label)))
    return np.vstack((lower_half, upper_half))


def bit_error_rate(
        targets: Union[torch.Tensor, np.ndarray],
        pred: Union[torch.Tensor, np.ndarray],
        bit_level: bool = False) -> np.float:
    """
    Get the bit error rate (BER) between the bits in `targets` and the bits in
    `pred`. This assumes gray-labled bits.
    :param targets: The bit sequence of the true symbols.
    :param pred: The predicted true symbols. If `bit_level` the shape (N,
        number bits per transmitted sample, i.e. 2 in case of PAM-4) is
        assumed, holding softbits. The hard bits are infered by applying a hard
        desicion: bit = 1 if softbit >= =, 0 else. In case of not `bit_level`
        the shape (N, number of possible symbols, i.e. len(alphabet)) is
        assumed, where each entry corresponds to a vote for the corresponding
        symbol. The bit corresponding to the maximum vote are then used to
        compute the BER.
    :returns: Returns the bit error rate.
    """
    if isinstance(targets, torch.Tensor):
        symbols = targets.detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().numpy()

    labels = get_graylabel(int(np.log2(pred.shape[1])))
    if bit_level:
        bits_hat = np.zeros_like(pred)
        bits_hat[pred >= 0] = 1
        return np.mean(symbols != bits_hat)
    bits = labels[symbols, :]
    pred = np.argmax(pred, 1)
    bits_hat = labels[pred, :]
    return np.mean(bits != bits_hat)


def apply_filter(
        inputs: torch.Tensor, filter: torch.Tensor, complex: bool = False) \
        -> torch.Tensor:
    """
    Applies a filter to inputs in the frequency domain and transforms back
    """
    conv = torch.from_numpy(
        np.fft.ifft(np.fft.fft(inputs.detach().numpy()) *
                    filter.detach().numpy()))
    return conv if complex else torch.real(conv).float()


def _raised_cosine(
        rolloff: float, period: int, freq: np.ndarray) -> np.ndarray:
    """ """
    h_f = np.cos(
        np.pi * period / 2 / rolloff
        * (np.abs(freq) - (1 - rolloff) / 2 / period)) ** 2
    idx = np.logical_and(
        (1 - rolloff) / 2 / period < np.abs(freq),
        np.abs(freq) <= (1 + rolloff) / 2 / period)
    filter_fd = np.zeros_like(freq)
    filter_fd[idx] = h_f[idx]
    idx1 = np.abs(freq) <= (1 - rolloff) / 2 / period
    filter_fd[idx1] = 1.
    return np.fft.fftshift(filter_fd)


def root_raised_cosine(
        signal_length: int, rolloff: float, samples_per_symbol: int) \
        -> torch.Tensor:
    """ """
    assert signal_length % 2 == 0, "signal_length must be even"
    freq = np.arange(-0.5, 0.5, 1 / signal_length)
    return torch.sqrt(torch.tensor(
        _raised_cosine(rolloff, samples_per_symbol, freq)).float())


def chromatic_dispersion(
        signal_length: int, sampling_freq: int, wavelength: int,
        dispersion_parameter: int, fiberlength: int) -> torch.Tensor:
    """ """
    freq = np.fft.fftfreq(signal_length) * sampling_freq
    light_speed = 3e8
    return torch.tensor(np.exp(
        1j * np.pi * wavelength**2 / light_speed * dispersion_parameter *
        fiberlength * freq**2))