# ARMOR: First-Order Masking of Activation and ArgMax Gadgets for Side-Channel Resistant Neural Networks

This artifact provides an implementation of our paper ARMOR: First-Order Masking of Activation and ArgMax Gadgets for Side-Channel Resistant Neural Networks accepted at TCHES 2026, Issue 2.

The artifact includes:
- Information loss analysis for Masked BNN (784-512-10)
- Clock cycle measurement for BNN (784-512-10)

## Requirements

Tested on:
- Ubuntu 22.04 LTS
- x86_64 CPU (Intel/AMD) supporting RDTSCP
- GCC (must support x86 intrinsics)

System dependencies:
- OpenSSL development package (libcrypto)

The MNIST dataset files required for our experiments can be obtained from the following repository:

https://github.com/takafumihoriuchi/MNIST_for_C

Please download the following files:

- `t10k-images.idx3-ubyte`(place this file in the `data/` directory)
- `t10k-labels.idx1-ubyte`(place this file in the `data/` directory)

---


## Running the Experiments

### Build
Compile the project with:

```bash
make
```

### Run
This artifact provides two evaluation modes:

- **Mean Clock Cycle Measurement** (`run_clock`)
- **Information Loss Calculation** (`run_accuracy`)

The desired mode is selected via a command-line argument:

- `0` → run_clock()
- `1` → run_accuracy()

Additionally, the user can select the MNIST dataset size, up to a maximum of 10,000 samples.

For example, to run the accuracy loss measurement with 10,000 MNIST samples illustrated in Table 4:

```bash
./test 1 10000
```

The expected result includes label loss, which measures whether the predicted label differs from the true label:
```bash
hidden layer node loss: 0
hidden layer loss: 0
output layer loss: 0
output loss: 0
label loss: 1097
```
