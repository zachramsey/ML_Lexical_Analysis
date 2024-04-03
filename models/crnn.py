import torch
import torch.nn as nn


"""
@brief  Model definition
@param  w_len:   maximum word length
"""
class Net(nn.Module):
    def __init__(self, w_len, dict_size):
        super(Net, self).__init__()

        self.embed_chars = nn.Embedding(dict_size, w_len, padding_idx=0)
        self.wGRU = nn.Sequential(nn.GRU(w_len, w_len, 2, batch_first=True, dropout=0.2, bidirectional=True))
        self.conv1 = nn.Sequential(
            nn.Conv1d(2*w_len, 8*w_len, 7),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 1) )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8*w_len, 32*w_len, 5),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 1) )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32*w_len, 64*w_len, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 1) )
        self.flatten = nn.Flatten()
        conv2_out = 256*w_len * ((((((w_len - 4) // 2) -3) // 2) - 2) // 2)
        self.dense1 = nn.Sequential(nn.Linear(conv2_out, 64*w_len), nn.ReLU(), nn.Dropout(0.5))
        self.dense2 = nn.Sequential(nn.Linear(64*w_len, 4*w_len), nn.ReLU(), nn.Dropout(0.5))
        self.output = nn.Sequential(nn.Linear(4*w_len, 1), nn.Sigmoid())

    """
    @brief  Forward pass through the model
    @param  w1: word1 input
            w2: word2 input
    @return x:  output of the model
    """
    def forward(self, w1, w2):
        # w1 = torch.tensor(w1, dtype=torch.long)
        # w2 = torch.tensor(w2, dtype=torch.long)
        # print(f'w1 inp: {w1.shape}')

        w1 = self.embed_chars(w1)
        w2 = self.embed_chars(w2)
        # print(f'w1 emb: {w1.shape}')

        w1, _ = self.wGRU(w1)
        w2, _ = self.wGRU(w2)
        # print(f'w1 gru: {w1.shape}')

        x = torch.cat((w1, w2), dim=1)
        # print(f' x cat: {x.shape}')

        x = self.conv1(x)
        # print(f' conv1: {x.shape}')

        x = self.conv2(x)
        # print(f' conv2: {x.shape}')

        x = self.conv3(x)
        # print(f' conv3: {x.shape}')

        x = self.flatten(x)
        # print(f'flatte: {x.shape}')

        x = self.dense1(x)
        # print(f'dense1: {x.shape}')

        x = self.dense2(x)
        # print(f'dense2: {x.shape}')

        x = self.output(x).squeeze(1)
        # print(f'output: {x.shape}')
        return x