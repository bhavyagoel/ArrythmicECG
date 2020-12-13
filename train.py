import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


optimizer = optim.Adam(network.parameters(), lr=1e-4, weight_decay=1e-5)

valid_loss_min = np.Inf

for epoch in range(200):

    train_loss = 0
    train_correct = 0

    val_loss = 0
    val_correct = 0

    #####TRAIN DATA SET#####
    network.train()
    for batch in train_loader:  # Get Batch

        labels, images = batch

        images = images.view(-1, 1, 3600).type(torch.cuda.FloatTensor)
        network.zero_grad()

        preds = network(images)  # Pass Batch
        labels = labels.type(torch.cuda.LongTensor)
        #         print(labels)
        #         print(preds)
        loss = nn.CrossEntropyLoss()(preds, labels)  # Calculate Loss

        #         optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        train_loss += loss.item()
        train_correct += get_num_correct(preds, labels)

    #####Validation DATA SET#####

    for batch in valid_loader:  # Get Batch

        labels, images = batch
        images = images.view(-1, 1, 3600).type(torch.cuda.FloatTensor)
        network.zero_grad()
        preds = network(images)  # Pass Batch
        labels = labels.type(torch.cuda.LongTensor)
        loss = nn.CrossEntropyLoss()(preds, labels)  # Calculate Loss

        #         optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        val_loss += loss.item()
        val_correct += get_num_correct(preds, labels)

    print("Epoch", epoch, " ;Total Train correct:", train_correct, " ;Train loss:", train_loss)
    print("Total Validation correct:", val_correct, " ;Validation Loss:", val_loss)

    # save model if validation loss has decreased
    if val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            val_loss))
        checkpoint = {'model': network,
                      'state_dict': network.state_dict(),
                      'optimizer': optimizer.state_dict()}

        torch.save(checkpoint, 'innovative_model/NewDeep_Attentive_BiModel_99.89.pth')
        valid_loss_min = val_loss

    network.eval()
    with torch.no_grad():
        correct_all = 0
        total_all = 0
        for labels, images in test_loader:
            #             print(images.shape)
            #             print(labels.shape)
            images = images.view(-1, 1, 3600).type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.FloatTensor)
            #             print(labels.size(0))
            preds = network(images)
            correct = 0
            total = 0

            for i in range(labels.size(0)):
                act_label = labels[i]  # act_label = 1 (index)
                pred_label = torch.argmax(preds[i])  # pred_label = 1 (index)

                #                 print(act_label)
                #                 print(pred_label)
                if (act_label == pred_label):
                    correct += 1
                total += 1
            total_all += total
            correct_all += correct
        #         print(correct_all)
        #         print(total_all)
        print('Test Accuracy of the model : {} %'.format(100 * (correct_all / total_all)))
    print('\n')
