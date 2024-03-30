import torch
import torch.utils.data as data
import pandas as pd


"""
@brief  Load data from csv file
        Split data into training, validation, and test sets
@param  file:       csv file containing data
        train_frac: fraction of data to use for training
@return train_ds:   training dataset
        eval_ds:    validation dataset
        test_ds:    test dataset
"""
class HypernymDataset(data.Dataset):
    def __init__(self, df):
        self.df = df

    """
    @brief  Get the length of the dataset
    @return len: length of the dataset
    """
    def __len__(self):
        return len(self.df)

    """
    @brief  Get the item at the specified index
    @param  idx: index of the item to retrieve
    @return w1:  word1 tensor
            w2:  word2 tensor
            l:   label tensor
    """
    def __getitem__(self, idx):
        w1 = torch.tensor(self.df.iloc[idx]['word1'], dtype=torch.float32)
        w2 = torch.tensor(self.df.iloc[idx]['word2'], dtype=torch.float32)
        l = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)
        return w1, w2, l


"""
@brief  Load data from csv file
        Split data into training, validation, and test sets
@param  file:       csv file containing data
        train_frac: fraction of data to use for training
        batch_size: size of each batch
@return train_dl:   training data loader
        eval_dl:    validation data loader
        test_dl:    test data loader
        w_len:      maximum word length
"""
def load_data(file, train_frac, batch_size):
    # Load data from csv file and encode to unicode integer representation
    df = pd.read_csv(file, names=['word1', 'word2', 'label'])
    df['word1'] = df['word1'].apply(lambda x: [ord(c) for c in x])
    df['word2'] = df['word2'].apply(lambda x: [ord(c) for c in x])

    # Determine the maximum length of words from either column
    w_len = max(df['word1'].apply(len).max(), df['word2'].apply(len).max())

    # Pad each word with zeros to make them all the same length
    df['word1'] = df['word1'].apply(lambda x: x + [0] * (w_len - len(x)))
    df['word2'] = df['word2'].apply(lambda x: x + [0] * (w_len - len(x)))

    # Split data into training, validation, and test sets
    train_data = df.sample(frac=train_frac, random_state=42)
    eval_data = df.drop(train_data.index).sample(frac=0.5, random_state=42)
    test_data = df.drop(train_data.index).drop(eval_data.index)

    train_ds = HypernymDataset(train_data)
    eval_ds = HypernymDataset(eval_data)
    test_ds = HypernymDataset(test_data)

    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_dl = data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, eval_dl, test_dl, w_len