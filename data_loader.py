import torch
import torch.nn as nn
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        w1 = torch.tensor(self.df.iloc[idx]['word1'], dtype=torch.long)
        w2 = torch.tensor(self.df.iloc[idx]['word2'], dtype=torch.long)
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
def load_data(files, train_frac, batch_size):
    # Compile data from multiple tsv files into a pandas dataframe
    print("Loading Data...")
    df = pd.concat([pd.read_csv(file, sep='\t') for file in files], ignore_index=True)
    df.drop_duplicates(inplace=True)    # Remove duplicate rows

    # Order frequency of characters in data from most common to least common
    char_freq = ''.join(df['word1'].values) + ''.join(df['word2'].values)
    char_freq = pd.Series(list(char_freq)).value_counts()
    char_freq = char_freq.sort_values(ascending=False)

    # Create a dictionary
    char_dict = {char: i+1 for i, char in enumerate(char_freq.index)}
    dict_len = len(char_dict) + 1

    # Map characters in data to integer indices
    df['word1'] = df['word1'].apply(lambda x: [char_dict[char] for char in x])
    df['word2'] = df['word2'].apply(lambda x: [char_dict[char] for char in x])

    # Pad char sequences to the maximum word length
    w_len = max(df['word1'].apply(len).max(), df['word2'].apply(len).max())
    df['word1'] = df['word1'].apply(lambda x: x + [0]*(w_len - len(x)))
    df['word2'] = df['word2'].apply(lambda x: x + [0]*(w_len - len(x)))

    # Split data into training, validation, and test sets
    train_data = df.sample(frac=train_frac, random_state=42)
    eval_data = df.drop(train_data.index).sample(frac=0.5, random_state=42)
    test_data = df.drop(train_data.index).drop(eval_data.index)

    # Create datasets
    train_ds = HypernymDataset(train_data)
    eval_ds = HypernymDataset(eval_data)
    test_ds = HypernymDataset(test_data)

    # Create dataloaders
    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_dl = data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, eval_dl, test_dl, w_len, dict_len