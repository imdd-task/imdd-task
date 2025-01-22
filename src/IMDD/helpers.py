""" Helper functions """
from typing import Union
import torch
import numpy as np


def get_graylabel(M_bits: int) -> torch.Tensor:
    if M_bits == 1:
        return torch.tensor([[0], [1]], dtype=torch.uint8)
    graybits = get_graylabel(M_bits - 1)
    lower = torch.hstack(
        (torch.zeros((2**(M_bits - 1), 1), dtype=torch.uint8), graybits))
    upper = torch.hstack(
        (torch.ones((2**(M_bits - 1), 1), dtype=torch.uint8),
        torch.flipud(graybits)))
    return torch.vstack((lower, upper))


def hard_decision(soft_bits: torch.Tensor) -> torch.Tensor:
    bits = torch.zeros_like(soft_bits)
    bits[soft_bits >= 0] = 1
    return bits


def accuracy(symbols: torch.Tensor, pred_symbols: torch.Tensor,
             bit_wise: bool = False) -> float:
    if bit_wise:
        pred = hard_decision(pred_symbols)
        return torch.count_nonzero((symbols != pred).sum(1) == 0) \
            / symbols.shape[0]
 
    return torch.count_nonzero(symbols == torch.argmax(pred_symbols, axis=1)) \
        / symbols.shape[0]


def bit_error_rate(targets: torch.Tensor, pred: torch.Tensor,
                   bit_level: bool = False) -> float:
    """
    Get the bit error rate (BER) between the bits in `targets` and the bits in
    `pred`. This assumes gray-labeled bits.
    :param targets: The bit sequence of the true symbols.
    :param pred: The predicted true symbols. If `bit_level` the shape (N,
        number bits per transmitted sample, i.e. 2 in case of PAM-4) is
        assumed, holding softbits. The hard bits are inferred by applying a hard
        decision: bit = 1 if softbit >= =, 0 else. In case of not `bit_level`
        the shape (N, number of possible symbols, i.e. len(alphabet)) is
        assumed, where each entry corresponds to a vote for the corresponding
        symbol. The bit corresponding to the maximum vote are then used to
        compute the BER.
    :returns: Returns the bit error rate.
    """
    labels = get_graylabel(
        int(torch.log2(torch.tensor(pred.shape[1])))).to(targets.device)
    if bit_level:
        bits_hat = torch.zeros_like(pred)
        bits_hat[pred >= 0] = 1
        return torch.mean(targets != bits_hat)
    bits = labels[targets, :]
    pred = torch.argmax(pred, 1)
    bits_hat = labels[pred, :]
    return torch.count_nonzero(bits != bits_hat) / bits.reshape(-1).shape[0]


def apply_filter(inputs: torch.Tensor, filter: torch.Tensor,
                 complex: bool = False) -> torch.Tensor:
    conv = torch.fft.ifft(torch.fft.fft(inputs) * filter)
    return conv if complex else torch.real(conv).float()


def raised_cosine(
        rolloff: float, period: int, freq: torch.tensor) -> torch.tensor:
    h_f = torch.cos(
        torch.pi * period / 2 / rolloff
        * (torch.abs(freq) - (1 - rolloff) / 2 / period)) ** 2
    idx = torch.logical_and(
        (1 - rolloff) / 2 / period < torch.abs(freq),
        torch.abs(freq) <= (1 + rolloff) / 2 / period)
    filter_fd = torch.zeros_like(freq)
    filter_fd[idx] = h_f[idx]
    idx1 = torch.abs(freq) <= (1 - rolloff) / 2 / period
    filter_fd[idx1] = 1.
    return torch.fft.fftshift(filter_fd)


def root_raised_cosine(signal_length: int, rolloff: float,
                       samples_per_symbol: int) -> torch.Tensor:
    assert signal_length % 2 == 0, "signal_length must be even"
    freq = torch.arange(-0.5, 0.5, 1 / signal_length)
    return torch.sqrt(raised_cosine(rolloff, samples_per_symbol, freq)).float()


def chromatic_dispersion(signal_length: int, sampling_freq: int,
                         wavelength: int, dispersion_parameter: int,
                         fiberlength: int) -> torch.Tensor:
    freq = torch.fft.fftfreq(signal_length) * sampling_freq
    light_speed = 3e8
    return torch.exp(
        1j * torch.pi * wavelength**2 / light_speed * dispersion_parameter *
        fiberlength * freq**2)
