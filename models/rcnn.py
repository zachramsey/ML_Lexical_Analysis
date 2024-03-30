import torch
import torch.nn as nn
import exception_handler as exc


"""
@brief  Model definition
@param  w_len:   maximum word length
"""
class Net(nn.Module):
    def __init__(self, w_len):
        super(Net, self).__init__()
        self.wGRU = nn.Sequential(nn.GRU(w_len, w_len, 2, batch_first=True, dropout=0.1, bidirectional=True))
        self.conv1 = nn.Sequential(
            nn.Conv1d(4*w_len, 8*w_len, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 1) )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8*w_len, 16*w_len, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 1) )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(nn.Linear(16*w_len, 8*w_len), nn.ReLU(), nn.Dropout(0.1))
        self.dense2 = nn.Sequential(nn.Linear(8*w_len, 4*w_len), nn.ReLU(), nn.Dropout(0.1))
        self.output = nn.Sequential(nn.Linear(4*w_len, 1), nn.Sigmoid())

    """
    @brief  Forward pass through the model
    @param  w1: word1 input
            w2: word2 input
    @return x:  output of the model
    """
    def forward(self, w1, w2):
        try:
            w1, _ = self.wGRU(w1)
            w2, _ = self.wGRU(w2)
            w1 = w1.unsqueeze(2)
            w2 = w2.unsqueeze(2)
            x = torch.cat((w1, w2), 1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.output(x).squeeze(1)
        except Exception: exc.print_tb()
        return x