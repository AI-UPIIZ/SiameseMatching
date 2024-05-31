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


## Structure of the Siamese Neural Network (First experiment):
1. Input Images: Two images (one selfie and one ID card) are fed into the network.
2. CNN (ResNet18): Both images pass through the same pre-trained ResNet18 CNN to extract features. The weights of this CNN are shared between the two branches to ensure consistent feature extraction.
3. Feature Embedding: The output of the CNN for each image is a feature vector (embedding) representing the image in a lower-dimensional space.
4. oncatenation: The two feature embeddings are concatenated to form a single combined vector.
5. Fully Connected Layers: The concatenated vector is passed through multiple fully connected layers to learn the relationship between the two embeddings.
6. Similarity Score: The final output layer produces a similarity score indicating whether the two images belong to the same person.

### Diagram
              Input Image 1 (Selfie)                Input Image 2 (ID Card)
                   +-------------+                       +-------------+
                   |   Selfie    |                       |   ID Card   |
                   +-------------+                       +-------------+
                         |                                     |
                         v                                     v
               +---------------------+               +---------------------+
               |     CNN (ResNet18)  |               |     CNN (ResNet18)  |
               |  (Shared Weights)   |               |  (Shared Weights)   |
               +---------------------+               +---------------------+
                         |                                     |
                         v                                     v
               +---------------------+               +---------------------+
               |   Feature Embedding |               |   Feature Embedding |
               +---------------------+               +---------------------+
                         |                                     |
                         +-----------------+-------------------+
                                           |
                                           v
                               +-------------------------+
                               |     Concatenation       |
                               +-------------------------+
                                           |
                                           v
                               +-------------------------+
                               | Fully Connected Layer 1 |
                               +-------------------------+
                                           |
                                           v
                               +-------------------------+
                               | Fully Connected Layer 2 |
                               +-------------------------+
                                           |
                                           v
                               +-------------------------+
                               | Fully Connected Layer 3 |
                               +-------------------------+
                                           |
                                           v
                               +-------------------------+
                               |    Similarity Score     |
                               +-------------------------+


