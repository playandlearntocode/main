'''
Multiclass classification example in PyTorch
Author:
Goran Trlin
https://playandlearntocode.com
'''

import torch # requires PyTorch
import time

# ---- start of settings:

EPOCHS = 2000 # number of training iterations
VERBOSE = True # render a lot of state info at each iteration. Set to False to improve performance.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# prefer cpu, due to the small model size
device = torch.device('cpu')

# ---- end of settings:

# classify points in a [-1,1]:[-1,1] 2D space into 4 classes, based on coordinates of each point

# classes / labels:
labels = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # top left corner
    [0.0, 1.0, 0.0, 0.0],  # top right,
    [0.0, 0.0, 1.0, 0.0],  # bottom left,
    [0.0, 0.0, 0.0, 1.0]  # bottom right,
])

# ---- start of input preparation
# input data:
# the points belong to the top left portion of the input 2D space
top_left_corner = torch.tensor([
    [-0.33, 0.22, 1.0, 0.0, 0.0, 0.0],
    [-0.53, 0.23, 1.0, 0.0, 0.0, 0.0],
    [-0.99, 0.24, 1.0, 0.0, 0.0, 0.0],
    [-0.95, 0.14, 1.0, 0.0, 0.0, 0.0],
    [-0.59, 0.54, 1.0, 0.0, 0.0, 0.0],
    [-0.99, 0.20, 1.0, 0.0, 0.0, 0.0]
])

top_right_corner = torch.tensor([
    [0.33, 0.22, 0.0, 1.0, 0.0, 0.0],
    [0.53, 0.23, 0.0, 1.0, 0.0, 0.0],
    [0.99, 0.24, 0.0, 1.0, 0.0, 0.0],
    [0.99, 0.24, 0.0, 1.0, 0.0, 0.0],
    [0.11, 0.33, 0.0, 1.0, 0.0, 0.0],
    [0.50, 0.45, 0.0, 1.0, 0.0, 0.0],
])

bottom_left_corner = torch.tensor([
    [-0.33, -0.22, 0.0, 0.0, 1.0, 0.0],
    [-0.53, -0.23, 0.0, 0.0, 1.0, 0.0],
    [-0.99, -0.24, 0.0, 0.0, 1.0, 0.0],
    [-0.25, -0.84, 0.0, 0.0, 1.0, 0.0],
    [-0.55, -0.54, 0.0, 0.0, 1.0, 0.0],
    [-0.35, -0.14, 0.0, 0.0, 1.0, 0.0]
])

bottom_right_corner = torch.tensor([
    [0.33, -0.22, 0.0, 0.0, 0.0, 1.0],
    [0.53, -0.23, 0.0, 0.0, 0.0, 1.0],
    [0.99, -0.24, 0.0, 0.0, 0.0, 1.0],
    [0.15, -0.09, 0.0, 0.0, 0.0, 1.0],
    [0.49, -0.43, 0.0, 0.0, 0.0, 1.0],
    [0.23, -0.25, 0.0, 0.0, 0.0, 1.0]
])

# all training and testing data combined:
data_all = torch.cat((top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner))

# randomize the sorting:
data_all = data_all[torch.randperm(len(data_all)), :]

# split into training and test subsets:
test_data_ratio = 0.2
test_ratio_start = int(data_all.shape[0] * (1 - test_data_ratio))

data_train = data_all[:test_ratio_start]
data_test = data_all[test_ratio_start:]

# training data:
y_train = data_train[:, 2:6]
x_train = data_train[:, :2]

# test data:
y_test = data_test[:, 2:6]
x_test = data_test[:, :2]

# ---- end of input preparation

class MulticlassClassifier(torch.nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features=2, out_features=10, device=dev)
        self.layer_2 = torch.nn.Linear(in_features=10, out_features=10, device=dev)
        self.layer_3 = torch.nn.Linear(in_features=10, out_features=4, device=dev)
        self.relu = torch.nn.ReLU()
        self.device = dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.layer_1(x))))

'''
Train the model on either CPU or GPU
'''
def train(model, loss_fn, epochs, x_train, y_train,labels, device, verbose):
    torch.seed()
    print(next(model.parameters()).device)
    model.train()  # enable gradient tracking
    for i in range(EPOCHS):
        y_logits = model(x_train)  # calculate predictions
        y_pred_probabilities = torch.softmax(y_logits, dim=1)

        loss = loss_fn(y_logits, y_train)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (verbose):
            print('Epoch: {}/{}'.format(i + 1, EPOCHS))
            print('Loss:')
            print(loss)
            print('Prediction:')
            print(y_pred_probabilities)

        lookup_values = (torch.argmax(y_pred_probabilities, dim=1))
        # print('lookup values:')
        # print(lookup_values)
        # print('raw predictions:')
        # print(y_pred_probabilities)

        # get reference values based on lookup array
        pred_labels = torch.index_select(labels, dim=0, index=lookup_values)

        if verbose:
            print('Predicted labels:')
            print(pred_labels)

        # print('Model weights:')
        # print(list(model.parameters()))
    model.eval()  # disable gradient tracking

'''
Run the trained model on the test dataset
'''
def test(model, loss_fn, x_test, y_test,labels, device, verbose):
    # test phase:
    y_test_pred_logits = model(x_test)
    y_test_pred_probabilities = torch.softmax(y_test_pred_logits, dim=1)

    # get reference values based on lookup array
    lookup_values = (torch.argmax(y_test_pred_probabilities, dim=1))
    pred_labels = torch.index_select(labels, dim=0, index=lookup_values)

    loss = loss_fn(y_test_pred_logits, y_test)
    if verbose:
        print('True test values:')
        print(y_test)

        print('-------------------')

        print('Predicted test values:')
        print(pred_labels)

    # correct predictions count:
    correct = torch.eq(torch.argmax(pred_labels, dim=1), torch.argmax(y_test, dim=1)).sum().item()
    total_tested = y_test.shape[0]

    if verbose:
        print('Accuracy on the test dataset:')
        print(str(correct * 1.0 / total_tested * 100) + '%')


# ---- start of main program
def main(epochs, verbose, device):

    # move the tensors to the selected device:
    local_labels = labels.to(device)
    local_x_train = x_train.to(device)
    local_y_train = y_train.to(device)

    local_x_test = x_test.to(device)
    local_y_test = y_test.to(device)

    model = MulticlassClassifier(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    start_time = time.time()

    train(model=model, loss_fn=loss_fn, epochs=epochs, x_train=local_x_train, y_train=local_y_train,labels=local_labels, device=device,
          verbose=verbose)
    test(model=model, loss_fn=loss_fn, x_test=local_x_test, y_test=local_y_test,labels=local_labels, device=device, verbose=verbose)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('Device used: {}'.format(device.type))
    print('Elapsed time: {} seconds '.format(round(elapsed_time, 2)))

# ---- end of main program
main(epochs=EPOCHS, verbose=VERBOSE, device=device)
