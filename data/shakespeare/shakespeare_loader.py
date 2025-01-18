from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    """
    Custom PyTorch Dataset for processing Shakespeare dialogues.
    Converts input data into sequences of indices for input and target processing.
    """
    def __init__(self, data, seq_len, n_vocab):
        """
        Initializes the ShakespeareDataset instance.

        Args:
            data: Dictionary containing dialogues (e.g., train_data or test_data).
            seq_len: Length of sequences to generate for the model.
            n_vocab: Size of the vocabulary for mapping characters to indices.
        """
        self.data = list(data.values())  # Convert the dictionary values to a list
        self.seq_len = seq_len  # Sequence length for the model
        self.n_vocab = n_vocab  # Vocabulary size

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of dialogues in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (input and target) from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple: Processed input (x) and target (y) tensors for the model.
        """
        dialogue = self.data[idx]  # Get the dialogue at the specified index
        x = process_x([dialogue], self.seq_len, self.n_vocab)[0]  # Prepare input tensor
        y = process_y([dialogue], self.seq_len, self.n_vocab)[0]  # Prepare target tensor
        return x, y