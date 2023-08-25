# Spiking Optical Communication
This code provides a PyTorch-based dataset for the training of demappers and
equalizers to compensate linear and non-linear impairments in an simulated
optical intesity modulation / direct detection (IM/DD) setup with pulse
amplitude modulation 4-level (PAM-4). This setup is described
[here](https://ieeexplore.ieee.org/abstract/document/10059327).

# The IM/DD Task

Given an sequence of messages $(a)^n = a^1, \dots, a^N$, where each message
$a^n$ corresponds to a bit sequence $[b_1,...b_M]^n$, that is transmitted
through an IM/DD link and meassured on the receiver side as
$(y)^n=y^1,\dots,y^N$, the task is to correct the received sequence $(y)^n$ in
such a way the bit error rate (BER) is minimized while keeping the energy and
memory footprint at a minimum..

The seqeunce $(y)^n$, is impaired lineraly due to chromatic dispersion in the
optical fiber, non-linearly by the photodiode, and gaussian noise. Because of
dispersion and bandwidth limitations, adjacent samples interfere such that
equalizing or demapping the sample $y^n$ to infere bits requires information
abouts its $n_\text{tap}/2$ predecessors and successors, hence the dataset
outputs for each $n$ the chuck $\mathcal{Y}^n = [y^{n - N_\text{tap}/2}, \dots,
y^n, \dots, y^{n + N_\text{tap}/2}]$.

In the case of PAM-4, each messages can assume four values, $a^n \in \lbrace 0,
1, 2, 3 \rbrace$ where each message is assigned to one of the (graylabled) bits
$[b_1b_2]^n \in \lbrace 00, 01, 11, 10 \rbrace$. Each chunk of samples
$\mathcal{Y}^n$ is then translated to a prediction of the send bits $[\hat{b}_1
\hat{b}_2]^n$. The BER is minimized by a demapper, receiving $\mathcal{Y}^n$
and outputting either predictions for the bits directly (bit-level, i.e.,
outputting softbits with subsequent hard desicion), or predictions for send
message (symbol-level, i.e., outputting a likelihood for each possible message
with subsequent bit assignment). The demapper is reaslized by an SNN. 

The dataset:
- Real-world application
- Competitive performance of SNNs to ANNs
- Clear metric: Minimize the bit error rate
- Infinite number of samples
- Guaranteed non-linearity
- Small input space and good results are achievable with relatively small
networks
- Interesting for benchmarking model efficiencies in terms of memory footprint
and computational complexity

_More inforamtion will follow soon ..._

# Using the Dataset
The dataset is used as any other PyTorch dataset. Its data can be provided by
`DataLoaders` as usual:
```python
# The parameters used in the IM/DD link
params = IMDDParams()
dataset = PAM4IMDD(params)

# Creating a data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

for i, (data, targets) in enmuerate(dataloader):
    # train ...
```

The dataset holds an instance of `IMDDModel` which simulates the IM/DD link.
The dataset randomly data, sends them through the link, and returns a chunk of
the (impaired) received samples in the shape `(n_taps)`, holding the the samples
$[y^{n - N_\text{tap}/2}, \dots, y^n, \dots, y^{n + N_\text{tap}/2}]$, for each
send data point $a^n$. It is recommended to enable shuffeling in the
`DataLoader` to avoid strongly corrrelated entreis in each batch. If all data
in the dataset is accessed, new data will be generated.