import torch.nn.functional as F

class CharLSTM(nn.Module):
    """
    Character-level LSTM model for text processing tasks.
    Includes embedding, LSTM, and a fully connected output layer.
    We use:
    - embedding size equal to 8;
    - 2 LSTM layers, each with 256 nodes;
    - densely connected softmax output layer.

    We can avoid to use explicitly the softmax function in the model and
    keep a cross entropy loss function as a loss function.

    as mentioned in paper [2] (Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
    Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan; Adaptive Federated Optimization, 2021)
    """
    def __init__(self, vocab_size = 70, embedding_size = 8, lstm_hidden_dim = 256, seq_length=80):
        super(CharLSTM, self).__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm1 = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass through the model.
        """
        # Layer 1: Embedding
        x = self.embedding(x)  # Output shape: (batch_size, seq_length, embedding_dim)

        # Layer 2: First LSTM
        x, _ = self.lstm1(x)  # Output shape: (batch_size, seq_length, lstm_hidden_dim)

        # Layer 3: Second LSTM
        x, hidden = self.lstm2(x)  # Output shape: (batch_size, seq_length, lstm_hidden_dim)

        # Layer 4: Fully Connected Layer
        x = self.fc(x)  # Output shape: (batch_size, seq_length, vocab_size)

        # Softmax Activation
        #x = self.softmax(x)  # Output shape: (batch_size, seq_length, vocab_size)
        return x[:, -1, :], hidden

    def init_hidden(self, batch_size):
        """Initializes hidden and cell states for the LSTM."""
        return (torch.zeros(2, batch_size, self.lstm_hidden_dim),
            torch.zeros(2, batch_size, self.lstm_hidden_dim))
        #2 is equal to the number of lstm layers!

