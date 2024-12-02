# Short-reach Optical Communications Datasets

Here, we describe an IM/DD task that is relevant to high-speed optical communication systems used in data centers.
Compared to other machine learning-inspired benchmarks, the task offers several advantages.
First, the dataset is inherently time-dependent, i.e., there is a time dimension that can be natively mapped to the dynamic evolution of SNNs.
Second, small-scale SNNs can achieve the target accuracy required by technical communication standards.
Third, due to the small scale and the defined target accuracy, the task facilitates the optimization for real-world aspects, such as energy efficiency, resource requirements, and system complexity.

The following description is widely taken from Arnold et al (2024), to which the reader is referred to for more details.

## Communications

<p align="center">
  <img src="figures/comm.png" />
</p>

The goal of a communication system is to transmit a binary message $\bm{b}\in\{0,1\}^{MN}$ of length $MN$ from a transmitter to a receiver over a physical channel without error.
To transmit the binary message, a group of ${M=\log_2|\mathcal{X}|}$ bits is mapped to a transmit symbol ${x\in\mathcal{X}}$, where ${\mathcal{X} \subset \mathbb{R}}$ denotes the set of transmit symbols.
Hence, transmission of $M \cdot N$ bits results in a sequence of $N$ transmits symbols ${\bm{x} \in \mathcal{X}^N}$.
The transmit symbols $\bm{x}$ are disturbed by the channel, resulting in the channel output ${\bm{y} \in \mathbb{C}^N}$. 
Based on the observation of $\bm{y}$, the receiver outputs an estimate of the transmit bit sequence $\hat{\bm{b}}$. The metric to classify the performance of the communication system is the BER:
```math
\mathrm{BER} = \frac{1}{MN} \sum_{n=0}^{MN-1} 1\left( b_n \neq \hat{b}_n \right) \quad \text{with}\quad 1\left( b_n \neq \hat{b}_n \right) = \begin{cases}
        0 \quad &\text{if} \quad b_n = \hat{b}_n\, , \\
        1 \quad &\text{if} \quad b_n \neq \hat{b}_n \\
    \end{cases} \, .
```

## Intensity modulation-direct detection (IM/DD) Link

<p align="center">
  <img src="figures/link.png" />
</p>

The IM/DD system is commonly used, e.g. in datacenters.
$M$ bits are grouped together and mapped to a real-valued transmit symbol $x$ using a pulse-amplitude modulation (PAM), where each of the ${m=2^M}$ symbols of the PAM represents an amplitude level.
In the tasks, PAM 4-level with $m=4$ intensity levels is used, and pairs of $M=\log_2 m = 2$ bits are mapped to one transmit symbol at index $q$ in $\mathcal{X}$.
We denote the bits mapped to $x[k]$ as $b_1[k]$ and $b_2[k]$:

| Symbol index $q$  | 1 | 2 | 3 | 4 |
|---|:-:|:-:|---|---|
| $b_1[k]\,b_2[k]$ | 00  | 01 | 11 | 10 |

The transmit sequence $\bm{x}$ is converted into a continuous-time signal, which is modulated onto the laser intensity.
Then, the modulated light is coupled to the optical fiber.
Inside the fiber, the frequency-dependent propagation speed results in chromatic dispersion (CD), and the symbols experience dispersion in the time domain.
Hence, consecutive symbols overlap and interfere, yielding the so-called inter-symbol interference (ISI).
The impact of CD on the $k$-th received symbol $y[k]$ can be narrowed down to the $n_\mathrm{ISI}$ neighboring symbols.
At the output of the optical fiber, the signal intensity is measured by a photo diode (PD), corresponding to a squaring of the electric field and introducing a non-linear distortion.
Additionally, thermal noise in the trans-impedance amplifier following the PD is modeled as AWGN with variance $\sigma_\mathrm{n}^2$.
In summary, we can write

```math
    y[k] = f\left( x\left[k -  \Big\lfloor \frac{n_\mathrm{ISI}}{2}  \Big\rfloor \right],\ldots,x[k],\ldots, x\left[k +  \Big\lfloor \frac{n_\mathrm{ISI}}{2}  \Big\rfloor \right] \right) + n[k] \quad \text{with} \quad n[k] \sim \mathcal{N}(0,\sigma_\mathrm{n}^2),
```

where $f(\cdot): \mathcal{X}^{n_{\mathrm{ISI}}}\rightarrow \mathbb{R}$ is an arbitrary function modeling both ISI and the non-linear distortion; $n[k]$ models the AWGN.

After the PD, the sequence of received values $\bm{y}$ is sampled over time.
The sequence $\bm{y}$ is passed to the receiver, which implements both channel equalization and demapping.
The receiver tries to mitigate the impact of the IM/DD link and recover the transmit bits $\bm{\hat{b}}$.
For a more detailed description of the implementation of the IM/DD link, the interested reader is referred to [1].

To ensure reliable transmission in practical systems, at the output of the receiver, a target BER is defined.
For a given link and noise power $\sigma_\mathrm{n}^2$, more powerful receivers can achieve a lower BER, resulting in a greater margin to the target BER.
This margin can be used to either realize higher data rates, transmit over longer fibers, or use less transmit power. 


# Tasks

We define two different parametrizations of the IM/DD link as separate tasks.
During training, the user is free to alter the parameters, e.g., the noise power $\sigma_\mathrm{n}^2$. 
However, during testing, only \texttt{n\_taps} can be chosen freely.


## LCD-Task

The low chromatic dispersion Task (LCD-Task) emphasizes non-linear impairment with little CD and thus weak ISI.
Receivers require a smaller number of $n_\text{taps}$ consecutive input samples or shorter memory, making it an excellent dataset for exploring resource efficiency and novel algorithms with small-scale networks.

### Parameters

| Parameter  | Value |
|---|:-:|
| `N` | 10000 |
| `n_taps` | 7 |
| `alphabet` | [-3, -1, 1, 3] |
| `oversampling_factor` | 3 |
| `baudrate` | 112 GBd |
| `wavelength` | 1270 mn |
| `dispersion_parameter` | -5 ps/nm/km |
| `fiber_length` | 4 km |
| `noise_power_db` | -20 dB |
| `roll_off` | 0.2 |
| `bias` | 2.25 |

## SSMF-Task
The standard single-mode fiber Task (SSMF-Task) models the standard single-mode fiber with a used wavelength of 1550 nm, which is widely used in practice due to its low attenuation and, thus, wide range.
The IM/DD link suffers from severe ISI, increasing the number of required input samples $n_\text{taps}$ and hence the need for more powerful equalizers.

### Parameters

| Parameter  | Value |
|---|:-:|
| `N` | 10000 |
| `n_taps` | 21 |
| `alphabet` | [0, 1, $\sqrt{2}$, $\sqrt{3}$] |
| `oversampling_factor` | 3 |
| `baudrate` | 50 GBd |
| `wavelength` | 1550 mn |
| `dispersion_parameter` | -17 ps/nm/km |
| `fiber_length` | 5 km |
| `noise_power_db` | -20 dB |
| `roll_off` | 0.2 |
| `bias` | 0.25 |



## Evaluation
To demonstrate the demapping performance, the achieved BER at a noise power of $\sigma_\mathrm{n}^2=-20\text{\,dB}$ should be reported.
The BER for various $\sigma_\mathrm{n}^2$ should be plotted, showing the ability of the receiver to generalize over a wide range of $\sigma_\mathrm{n}^2$. 
Incrementing the noise power $\sigma_\mathrm{n}^2$ by $1\text{\,dB}$ provides a sufficiently fine resolution.
To ensure statistical significance of the reported BER, for each $\sigma_\mathrm{n}^2$ and thus BER data point, new data should be transmitted until a minimum of 2000 bit error events are encountered.
To measure the complexity of the model and anticipate its energy consumption, the number of trainable parameters and, in the case of SNN, the average number of spikes sent to the SNN and the spikes emitted by the neurons within the SNN per processing of one symbol are to be reported.


## Using the Dataset

The PyTorch-based datasets corresponding to the defined tasks are imported as `LCDDataset` and `SSMFDataset`.

```python
from torch.utils.data import DataLoader
from IMDD import LCDDataset
# Dataset
dataset = LCDDataset(bit_wise=False)
dataset.set_n_taps(n_tap)
# Data loader
dataloader = DataLoader(dataset, batch_size, shuffle=True)
for (y_chunk, q) in dataloader:
    ... # train
```

After completing one epoch (i.e., accessing all $\texttt{N}$ samples in the dataset), the dataset generates new data instead of reusing the same transmit samples, which can be disabled by setting `continuous_sampling=False`.
The number of taps $n_\text{taps}$ and the noise power $\sigma_\text{n}^2$ are set by via `set_n_taps` and `set_noise_power_db`.


An `IMDDModel` in the dataset generates indices $q[k]$ of transmit symbols $\boldsymbol{x}$ in $\mathcal{X}$ and outputs the received chunks $\boldsymbol{y}^\text{c}[k]$, from which the pairs of bits $\hat{b}_1[k]\hat{b}_2[k]$ are estimated.
The dataset returns a chunk tensor at index `k` of shape `(n_taps,)`, representing the chunked received data together with the index $q[k]$ of shape `(1,)`.
Hence, the tensor holds all symbols required to process $y[k]$ at position `n_taps//2`.
This allows shuffling and batching along the first dimension using a PyTorch `Dataloader`, resulting in `y_chunk` of shape `(batch_size, n_taps)` and `q` of shape `(batch_size,)`. 
Each transmit symbol is labeled with its assigned bits, generated using the provided `get_graylabels` function, and accessed at the corresponding index `q`.
By default, the dataset does not directly return bit-level labels to maintain flexibility.
This design supports symbol-level receivers, allowing models to output class index $\hat{q}$ corresponding to the predicted symbol $\hat{x} \in \mathcal{X}$, which can be used together with $q$ in a cross-entropy loss function.
For tasks requiring bit-level outputs, setting `bit_level=True` in the dataset configuration changes the format of $\bm{q}$ to `(N, 2)`, where each entry contains binary values corresponding to the respective bits.
To compute the BER based on either predicted symbols or bits, we provide a helper function `bit_error_rate`.

An arbitrarily parameterized IM/DD link can be created by passing an instance of `IMDDParams` to an `IMDDModel`.
A dataset is created by instantiating `IMDDDataset` with the given parameters.
This allows users to adapt the provided link implementations to suit their specific requirements.

```python
from IMDD import IMDDDataset, IMDDParams
# Parameterization
params = IMDDParams(...)
# Link model
link = IMDDModel(params)
# Dataset
dataset = IMDDDataset(params, bit_level=False)
```

# Citation
If you use this dataset, please cite as:
```python
...
```

# References

[1] E. Arnold, G. Böcherer, F. Strasser, E. Müller, P. Spilger, S. Billaudelle, et al., “Spiking neural network nonlinear demapping on neuromorphic hardware for IM/DD optical communication,” Journal of Lightwave Technology, vol. 41, no. 11, pp. 1–8, 2023. DOI: 10.1109/JLT.2023. 3252819.