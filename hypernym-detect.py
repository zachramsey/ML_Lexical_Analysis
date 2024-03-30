"""
Sp24 | ECE:5995 AML | Midterm Project
Zach Ramsey
Date: 03-18-2024

Description:
* Task:         Given two words, determine if the first word is a hypernym of the second word
* Features:     Two words represented as unicode code points
* Labels:       Boolean value indicating if the first word is a hypernym of the second word
* Model:        Convolutional Neural Network (CNN) or Recurrent Convolutional Neural Network (RCNN)
* Loss:         Binary Cross Entropy Loss
* Optimizer:    Adam
* Testing:      Accuracy of model on test set (%)
"""

import time
import torch
import torch.nn as nn

import data_loader as dl
import models.cnn as cnn
import models.rcnn as rcnn


"""
@brief Train the model
@param  net:        model
        criterion:  loss function
        optimizer:  optimization algorithm
        data:       training data
@return net:        trained model
"""
def train(net, device, criterion, optimizer, train_data, eval_data):
    train_loss = 0
    eval_loss = 0
    eval_losses = []

    # Train model for limited number of epochs
    print(f'Train Batches: {len(train_data)} | Eval Batches: {len(eval_data)}')
    start = time.time()
    for epoch in range(50):

        # Train model on training set
        running_loss = 0
        epoch_start = time.time()
        for i, data in enumerate(train_data):
            w1, w2, label = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            output = net(w1, w2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 0:
                print(f'    ({round(time.time() - epoch_start, 1)} s) | Batch: {i} | Train Loss: {round(running_loss / (i+1), 8)}', end='\r')
        train_loss = running_loss/len(train_data)

        # Evaluate model on validation set
        running_loss = 0
        for data in eval_data:
            w1, w2, label = data[0].to(device), data[1].to(device), data[2].to(device)
            output = net(w1, w2)
            loss = criterion(output, label)
            running_loss += loss.item()
        eval_loss = running_loss/len(eval_data)
        eval_losses.append(eval_loss)

        # Print losses for each epoch
        print(f'    ({round(time.time() - epoch_start, 1)} s) | Epoch: {epoch} | Train Loss: {round(train_loss, 8)} | Eval Loss: {round(eval_loss, 8)}')

        # Early stopping if validation loss increases
        check_prior_losses = 3
        avg_last_losses = sum(eval_losses[-check_prior_losses-1:-1])/check_prior_losses
        if (len(eval_losses) > 5) and (eval_losses[-1] > avg_last_losses): break

    print(f'Training Time: {round(time.time() - start, 1)} s')
    return train_loss, eval_loss


"""
@brief Test the model
@param  net:        model
        criterion:  loss function
        data:       test data
@return avg_loss:   average loss
"""
def test(net, device, criterion, test_data):
    running_loss = 0
    correct = 0
    total = 0

    print(f'Test Batches: {len(test_data)}')
    start = time.time()
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data):
            w1, w2, label = data[0].to(device), data[1].to(device), data[2].to(device)
            output = net(w1, w2)

            loss = criterion(output, label)
            running_loss += loss.item()

            for i in range(len(output)):
                total += 1
                output[i] = 1 if output[i] > 0.5 else 0
                correct += 1 if output[i] == label[i] else 0

    avg_loss = running_loss/total
    accuracy = correct/total
    print(f'    Test Loss: {round(running_loss / (i+1), 8)} | Accuracy: {round(correct/total, 6)*100}%', end='\r')
    print(f'\nTest Time: {round(time.time() - start, 1)} s')
    return avg_loss, accuracy


"""
@brief Main function
"""
def main():
    start = time.time()
    print("----------------------------------------------------------------")
    datafile = 'data\compiled_data.csv'
    train_frac = 0.8
    batch_size = 64
    learning_rate = 0.0002
    weight_decay = 2e-5
    print(f'Train Frac: {train_frac} | Batch Size: {batch_size} | LR: {learning_rate} | WD: {weight_decay}')

    # Load data and create dataloaders
    train_data, eval_data, test_data, w_len = dl.load_data(datafile, train_frac, batch_size)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = cnn.Net(w_len).to(device)
    # net = rcnn.Net(w_len).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train and test model
    print("----------------------------------------------------------------")
    train_loss, eval_loss = train(net, device, criterion, optimizer, train_data, eval_data)
    print("----------------------------------------------------------------")
    test_loss, accuracy = test(net, device, criterion, test_data)

    # Print results
    print("----------------------------------------------------------------")
    print(f'Train Loss: {round(eval_loss, 8)}')
    print(f' Eval Loss: {round(train_loss, 8)}')
    print(f' Test Loss: {round(test_loss, 8)}')
    print(f'  Accuracy: {round(accuracy, 6)*100}%')
    print(f'Total Time: {round(time.time() - start, 1)} s')
    print("----------------------------------------------------------------")

if __name__ == '__main__':
    main()
