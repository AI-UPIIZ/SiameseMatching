# Deep Learning Approach for Person Matching in Images: A Siamese Network Perspective

## Abstract
This repository contains the implementation and research for a Siamese Neural Network aimed at identifying whether two given images represent the same person. The project explores deep learning techniques for person re-identification and verification, leveraging the unique architecture of Siamese Networks to learn discriminative feature representations. The network is trained to minimize the distance between feature vectors of images of the same person while maximizing the distance for different individuals. This work includes experiments, evaluations, and comprehensive documentation to provide insights into the effectiveness of Siamese Networks in person matching tasks. Our results demonstrate significant improvements in accuracy and robustness in person identification, contributing to the advancement of facial recognition technologies.

## Project Structure


## Pre Requisites
- Python 3.9+
- Astral UV


## Installation of dependencies
In order to install all necessary packages to run experiments, execute the following:

### Virtual environment

```bash
uv venv .venv-siamese-matching
```

### Activation of the virtual environment

```bash
source ./.venv-siamese-matching/bin/activate
```

### Installation of dependencies

```bash
make install
```

