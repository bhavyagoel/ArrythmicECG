import numpy as np
import torch
import torch.nn.functional as F


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


network_best = load_checkpoint('./NewDeep_Attentive_BiModel_99.89')
network_best = network_best.cuda()
print(network_best)

# accuracy_train= (train_correct/len(train))*100
# print('Train Accuracy=' + str(accuracy_train) + '%')


test_loss = 0.0
class_correct = list(0. for i in range(20))
class_total = list(0. for i in range(20))

network_best.eval()  # prep model for evaluation

for target, data in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    data = data.view(-1, 1, 3600).type(torch.cuda.FloatTensor)
    output = network_best(data)
    # calculate the loss
    loss = F.cross_entropy(output, target)
    # update test loss 
    test_loss += loss.item() * data.size(0)
    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)
    # compare predictions to true label

    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss / len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(20):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %.6f%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        continue

print('\nTest Accuracy (Overall): %.6f%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
