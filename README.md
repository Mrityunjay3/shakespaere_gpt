# ShakesPaere GPT

## Overview

Welcome to ShakesPaere GPT. This language model has been made following the amazing lectures curated and delivered by Andres Karpathy (Co-founder of OpenAI). This repository contains a Language Model built using PyTorch, specifically designed to generate text that emulates the unique literary style of Shakespeare. The model leverages a transformer architecture, renowned for its effectiveness in natural language processing tasks. The model is trained on the dataset "input.txt" which is a compilation of all of Shakespeare's work (popularly known as tiny Shakespeare).

## Technical Specifications

- **Programming Language**: Python 3.x
- **Framework**: PyTorch, with optional CUDA support for enhanced processing on GPUs.
- **Model Architecture**:
  - **Transformer Base**: Utilizes a custom transformer architecture with multi-headed self-attention mechanisms to manage long-range dependencies within text.
  - **Number of Parameters**: The model includes approximately 12 million trainable parameters, optimized for high-performance text generation.
  - **Components**:
    - **Multi-headed Self-attention**: Analyzes different segments of the input sequence to predict subsequent characters accurately.
    - **Feedforward Neural Networks**: Further refines the outputs from the attention modules.
    - **Normalization and Residual Connections**: Ensures stable learning and integrates different learning phases effectively.

## Installation

Clone this repository to set up the project locally:

```bash
git clone https://github.com/yourusername/shakespaere_gpt.git
cd shakespaere_gpt
```

## Usage

Execute the following command to start the training process and generate text in the style of Shakespeare:

```bash
python train.py
```

## Model Details

- **Token and Position Embeddings**: Converts textual input into a format suitable for neural network processing.
- **Architecture Details**:
  - The model is structured around several layers of a transformer-based architecture, each consisting of a multi-headed self-attention mechanism followed by a feedforward network.
  - Layer normalization and dropout are employed within each transformer block to regulate training and prevent overfitting.

## Training and Evaluation

- **Batch Processing**: Handles large volumes of data in manageable batches for efficient training.
- **Loss Metrics**: Utilizes cross-entropy loss to measure and optimize the accuracy of predictions.
- **Regular Validation**: Periodically evaluates the model against a validation dataset to ensure it generalizes well beyond the training data.

---
