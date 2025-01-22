""" IM/DD channel model with PAM-4 ASK """
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass
import torch

from IMDD.helpers import root_raised_cosine, apply_filter, chromatic_dispersion


@dataclass
class IMDDParams:
    """
    :param N: The number of symbols to transmit.
    :param n_taps: The number of taps used in the demultiplexer (and demapper).
    :param alphabet: The used alphabet. Each message that is send is mapped to
        one element in the alphabet.
    :param oversampling_factor: The factor by which to up-sample, resp.
        down-sample the traces in the transmitter, resp. receiver.
    :param baudrate: The symbol rate in GBd. This corresponds to the rate with
        which the transmitter sends the symbols through the link, resp. the
        receiver samples the incoming symbols.
    :param wavelength: The carrier wave's wave length in nm. This value is used
        to compute the chromatic dispersion.
    :param dispersion_parameter: Dispersion parameter of the optical fiber
        material, determining the dispersion, given in value ps / nm / km.
    :param fiber_length: The assumed length of the optical fiber in km.
    :param noise_power_db: The assumed SNR in optical link in dB.
    :param roll_off: The roll-off factor used in the root-raised cosine filter.
    """
    N: int = 10000
    n_taps: int = 7
    alphabet: torch.Tensor = torch.tensor([-3., -1., 1., 3.])
    oversampling_factor: int = 3
    baudrate: int = 112
    wavelength: float = 1270
    dispersion_parameter: float = -5
    fiber_length: int = 4
    noise_power_db: float = -20.
    roll_off: float = 0.2
    bias: Optional[float] = 2.25


@dataclass
class LCDParams(IMDDParams):
    """ IM/DD parameters for LCD-Task """
    n_taps: int = 7
    alphabet: torch.Tensor = torch.tensor([-3., -1., 1., 3.])
    baudrate: int = 112
    wavelength: float = 1270.
    dispersion_parameter: float = -5
    fiber_length: int = 4
    bias: Optional[float] = 2.25


@dataclass
class SSMFParams(IMDDParams):
    """ IM/DD parameters for SSMF-Task """
    n_taps: int = 21
    alphabet: torch.Tensor = torch.tensor(
        [0., 1., torch.sqrt(torch.tensor(2.)), torch.sqrt(torch.tensor(3.))])
    baudrate: int = 50
    wavelength: float = 1550.
    dispersion_parameter: float = -17
    fiber_length: int = 5
    bias: Optional[float] = 0.25


class Transmitter(torch.nn.Module):
    """ Class implementing the transmitter model """

    def __init__(self, params: IMDDParams):
        """
        :param params: Parameter object prameterizing the IM/DD model.
        """
        super().__init__()
        self.params = params
        self.filter = root_raised_cosine(
            params.N * params.oversampling_factor, params.roll_off,
            params.oversampling_factor)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The transmitter gets the message sequence (as integers) to be send
        through the optical link. The bits are mapped to their corresponding
        symbol in the given alphabet, up-sampled, filtered with a root-raised
        cosine filter, biased and normalized. This implements the E/O, i.e. the
        modulation onto the carrier wave.

        :param input: The integer sequence of shape (N,) to be send. Each entry
            is mapped to one symbol in the alphabet and represents the bits to
            be transmitted.
        :returns: Returns the sequence of values corresponding to the modulated
            light pulses that are send through the fiber.
        """
        symbols = self.params.alphabet[input]

        symbols_up = torch.zeros(
            self.params.oversampling_factor * len(symbols))
        symbols_up[::self.params.oversampling_factor] = symbols

        tx = apply_filter(symbols_up, self.filter)

        if self.params.bias is None:
            tx_biased = tx + torch.abs(torch.min(tx))
        else:
            tx_biased = tx + self.params.bias

        tx_normed = 1. / torch.sqrt(torch.mean(tx_biased**2)) * tx_biased

        return tx_normed


class OpticalChannel(torch.nn.Module):
    """ Class implementing the optical fiber channel model """

    def __init__(self, params: IMDDParams, rng: torch.Generator):
        """
        :param params: Parameter object prameterizing the IM/DD model.
        :param rng: Random generator to generate noise.
        """
        super().__init__()
        self.params = params
        self.rng = rng
        self.cd_filter = chromatic_dispersion(
            params.oversampling_factor * params.N,
            params.oversampling_factor * params.baudrate * 1e9,
            params.wavelength * 1e-9, params.dispersion_parameter * 1e-6,
            params.fiber_length * 1e3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The channel gets the symbols to be send from the transmitter and then
        adds chromatic dispersion. The measurement by the photo diode is
        implemented by a absolute square on the output of the fiber.

        :param input: The sequence of symbols to be send through the fiber.
        :returns: The symbols measured by the photo diode.
        """
        y_cd = apply_filter(input, self.cd_filter, complex=True)
        y_pd = torch.abs(y_cd)**2
        noise_power = torch.sqrt(
            torch.tensor(10**(self.params.noise_power_db / 10)))
        y_wgn = y_pd + noise_power * torch.randn(y_pd.shape, generator=self.rng)
        return y_wgn


class Receiver(torch.nn.Module):
    """ Class implementing the receiver model """

    def __init__(self, params: IMDDParams) -> None:
        """
        :param params: Parameter object prameterizing the IM/DD model.
        """
        super().__init__()
        self.params = params
        self.filter = root_raised_cosine(
            params.N * params.oversampling_factor, params.roll_off,
            params.oversampling_factor)

    def forward(self, input: torch.Tensor, disable_down: bool = False) \
            -> Tuple[torch.Tensor]:
        """
        The receiver applies a root-raised-cosine and down-samples the signal
        for simulating the sampling of the electrical signal after the O/E.

        :param input: The signal form the optical channel.
        :returns: The sampled signal multiplexed into shape (N, n_taps)
        """
        rx = apply_filter(input, self.filter)

        if disable_down:
            return rx

        rx_down = self.params.oversampling_factor \
            * rx[::self.params.oversampling_factor]
        rx_down = rx_down.reshape(-1, 1)

        rx_dmux = torch.zeros(
            rx_down.shape[0], self.params.n_taps, dtype=rx_down.dtype)
        for j in range(self.params.n_taps):
            rx_dmux[:, j:j + 1] = torch.roll(
                rx_down, j - self.params.n_taps // 2)

        return rx_dmux


class IMDDModel(torch.nn.Module):
    """
    IM/DD model consisting of a Transmitter, optical Channel, and Receiver
    """

    def __init__(self, params: IMDDParams, seed: int = 0,
                 device: torch.device = "cpu"):
        """
        :param params: Parameter object holding the model's parameters.
        :param seed: Random seed to random generator for sampling symbols to
            send and generating noise in the link.
        :param device: The device to operate on. Either "cpu" or "cuda".
        """
        super().__init__()
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.params = params
        self.transmitter = Transmitter(params)
        self.channel = OpticalChannel(params, self.rng)
        self.receiver = Receiver(params)

    def extra_repr(self):
        """ Model parameterization """
        return str(self.params)

    def source(self) -> torch.Tensor:
        """
        Generate a sequence of messages to send through the channel.

        :returns: A tensor containing a the indices of the message in the used
            alphabet randomly drawn.
        """
        p = torch.ones(len(self.params.alphabet)) / len(self.params.alphabet)
        message = p.multinomial(
            self.params.N, replacement=True, generator=self.rng)
        return message

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Sends a sequence of messages through the IM/DD channel model.
    
        :param input: The sequence of messages to be transmitted through the
            model.
        :returns: Returns the sequence of the received symbols by the model's
            receiver.
        """
        data = self.transmitter(input)
        data = self.channel(data)
        data = self.receiver(data)
        return data