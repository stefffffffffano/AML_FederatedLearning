# Shakespeare Character-Level LSTM Model

## Dataset Description

The Shakespeare dataset used in this project is tailored for federated learning applications and character prediction tasks. The dataset consists of concatenated lines from the plays written by William Shakespeare, providing a rich vocabulary of old English text. Each line in the dataset has been treated as a separate data point, facilitating character-level sequence modeling.

The dataset was borrowed from the Leaf project[1].

### Data Structure
- **users**: Unique identifiers for different characters from various plays who have dialogue.
- **num_samples**: The number of samples (lines of text) corresponding to each user.
- **user_data**: A dictionary where each entry contains sequences of text spoken by the character.

### Input and Output
- **x**: Input sequences; each sequence is a string of characters from a line of dialogue.
- **y**: The target sequence for prediction; typically the next character in the line after the given sequence, allowing the model to learn character-level predictions.

## Model Architecture: CharLSTM

This repository includes the implementation of a Character-Level LSTM (Long Short-Term Memory) model designed to predict the next character in a sequence of Shakespeare's text.

The model was borrowed from "Adaptive Federated Optimization" [2].

### Components
- **Embedding Layer**: Converts character indices to dense vector representations with an embedding size of 8.
- **LSTM Layers**: Two LSTM layers, each with 256 nodes, process the sequence of embeddings.
- **Fully Connected Output Layer**: Maps the output of the last LSTM layer to a vector with a size equal to the vocabulary, representing the probability distribution over possible next characters.

### Model Specifications
- **Vocabulary Size**: 70 (characters in the dataset)
- **Embedding Size**: 8
- **LSTM Hidden Dimension**: 256
- **Sequence Length**: 80 (characters)

### Forward Pass
1. **Embedding**: The input sequence is first converted into embeddings.
2. **First LSTM**: Processes the sequence of embeddings.
3. **Second LSTM**: Further processes the output from the first LSTM.
4. **Output Layer**: The last LSTM output is transformed into logits for each character in the vocabulary.

The final output of the model is the logits for the last character of the input sequence, used to predict the next character.

### Initialization
- The model initializes hidden and cell states to zero tensors at the start of each batch, necessary for stateful LSTM operation.

## Usage

The model can be used in a federated learning setting where each user's data serves as a local dataset for training separate model instances. The character prediction capabilities of this model are tested by training on sequences from the Shakespeare dataset and predicting the next characters.

# Training and Evaluation
- The model is trained using cross-entropy loss to compare the predicted character distribution with the true next character.

## Hyperparameters
Since the leaf implementation of the pre-processing for the dataset doesn't provide any validation split, the hyperparameters were selected using the one indicated in the paper [2] used as a reference for this work.

## References
[1] Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." arXiv preprint arXiv:1812.01097 (2018).

[2] Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan; "Adaptive Federated Optimization," 2021.
