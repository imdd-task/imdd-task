""" IM/DD channel model with PAM-4 ASK """

from typing import Tuple, Optional, NamedTuple
import numpy as np
import torch

from .helpers import root_raised_cosine, apply_filter, chromatic_dispersion


class IMDDParams(NamedTuple):
    """
    :param N: The number of symbols to transmit.
    :param n_taps: The number of taps used in the demultiplexer (and demapper).
    :param alphabet: The used alphabet. Each message that is send is mapped to
        one element in the alphabet.
    :param oversampling_factor: The factor by which to up-sample, resp.
        down-sample the traces in the transmitter, resp. receiver.
    :param baudrate: The symbol rate in Hz. This corresponds to the rate with
        which the transmitter sends the symbols through the link, resp. the
        receiver samples the incoming symbols.
    :param wavelenght: The carrier wave's wave length. This value is used to
        compute the chromatic dispersion.
    :param dispersion_parameter: Dispersion parameter of the optical fiber
        material, determing the dispersion, given in value s/m**2.
    :param fiber_lenght: The assumed length of the optical fiber in m.
    :param noise_power_gain_db: The assumed SNR in optical link in dB.
    :param roll_off: The roll-off factor used in the root-raised cosine filter.
    """
    N: int = 10000
    n_taps: int = 7
    alphabet: torch.Tensor = torch.tensor([-3., -1., 1., 3.])
    oversampling_factor: int = 3
    baudrate: int = 112e9
    wavelength: float = 1270e-9
    dispersion_parameter: float = -5e-6
    fiber_length: int = 4e3
    noise_power_gain_db: float = 20.
    roll_off: float = 0.2
    bias: Optional[float] = 2.25


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
        symbol, up-sampled, filtered wiht a root-raised cosine filter, biased
        and normalized. This implements the E/O, i.e. the modulation onto the
        carrier wave..
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

    def __init__(self, params: IMDDParams):
        """
        :param params: Parameter object prameterizing the IM/DD model.
        """
        super().__init__()
        self.params = params
        self.cd_filter = chromatic_dispersion(
            params.oversampling_factor * params.N,
            params.oversampling_factor * params.baudrate, params.wavelength,
            params.dispersion_parameter, params.fiber_length)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The channel gets the symbols to be send from the transmitter and then
        adds chromatic dispersion. The measurement by the photodiode is
        implemented by a absolute square on the output of the fiber.
        :param input: The sequence of symbols to be send through the fiber.
        :returns: The symbols measured by the photodiode.
        """
        y_cd = apply_filter(input, self.cd_filter, complex=True)
        y_pd = torch.abs(y_cd).float()**2
        noise_power = torch.sqrt(
            torch.tensor(10**(-self.params.noise_power_gain_db / 10)))
        y_wgn = y_pd + noise_power * torch.randn(y_pd.shape)
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
        for simulating the sampling od the electrical signal after the O/E.

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

    def __init__(self, params: IMDDParams):
        """
        :param params: Parameter object holding the model's parameters.
        """
        super().__init__()
        self.params = params
        self.transmitter = Transmitter(params)
        self.channel = OpticalChannel(params)
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
        message = torch.tensor(np.random.choice(
            range(len(self.params.alphabet)), self.params.N)).long()
        return message

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Send a sequence of messages through the IM/DD channel model.
        :param data: The sequence of messanges to be transmitted through the
            model.
        :returns: Returns the sequence of the received symbols by the model's
            receiver.
        """
        data = self.transmitter(input)
        data = self.channel(data)
        data = self.receiver(data)
        return data