'''
--------------------------------
Sp24 | ECE:5995 | Midterm Project
Zach Ramsey
Date: 03-18-2024
--------------------------------
Hypernym Detection
Split data into training, validation, and test sets
Model:
    Input Layer (word1 & word2 tokenized by character)
    CNN Layers
    GRU Cells
    Fully Connected Layers
    Ouput Layer (Binary Probability of Hypernymy)
--------------------------------
'''

import pandas as pd

def data_parser(file, train_frac):
    df = pd.read_csv('your_data.csv', names=['word1', 'word2', 'label'])
    d_train = df.sample(frac=train_frac, random_state=42)
    d_val = df.drop(d_train.index).sample(frac=0.5)
    d_test = df.drop(d_train.index).drop(d_val.index)
    return d_train, d_test, d_val      

def main():
    datafile = 'data/complied_data.csv'
    train_frac = 0.8
    d_train, d_test, d_val = data_parser(datafile, train_frac)

if __name__ == '__main__':
    main()