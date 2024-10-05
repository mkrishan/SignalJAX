import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

OPTIMIZERS = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adagrad': optim.Adagrad
}


class ModelDM:

    def __init__(self, lr=0.005, img_rows=28, img_cols=28, img_channels=1, n_l1=16, n_l2=None, dout=0, n_filt=8, opt='Adam',
                 regularizer=None, verbose=False, seed=1337, loss='cross_entropy'):

        self.lr = lr
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.n_l1 = n_l1
        self.n_l2 = n_l2
        self.dout = dout
        self.n_filt = n_filt
        self.opt = OPTIMIZERS[opt]
        self.regularizer = regularizer
        self.verbose = verbose
        self.seed = seed
        self.loss = loss

        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def create_model(self):
        torch.manual_seed(self.seed)

        layers = [
            nn.Conv2d(self.img_channels, self.n_filt, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.n_filt * ((self.img_rows - 2) // 2) * ((self.img_cols - 2) // 2), self.n_l1),
            nn.ReLU()
        ]

        if self.dout != 0:
            layers.append(nn.Dropout(self.dout))

        if self.n_l2 is not None:
            layers.append(nn.Linear(self.n_l1, self.n_l2))
            layers.append(nn.ReLU())

            if self.dout != 0:
                layers.append(nn.Dropout(self.dout))

        layers.append(nn.Linear(self.n_l2 or self.n_l1, 10))  # Output layer for 10 classes
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

        if self.verbose:
            print(self.model)

        self.optimizer = self.opt(self.model.parameters(), lr=self.lr)
        if self.loss == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()

    def train(self, X_train, Y_train, X_test, Y_test, batch_size, nb_epoch):
        X_train, Y_train = torch.Tensor(X_train), torch.Tensor(Y_train).long()
        X_test, Y_test = torch.Tensor(X_test), torch.Tensor(Y_test).long()

        train_data = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for epoch in range(nb_epoch):
            self.model.train()
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, X, Y):
        X, Y = torch.Tensor(X), torch.Tensor(Y).long()
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            loss = self.loss_fn(output, Y)
            correct = (output.argmax(dim=1) == Y).sum().item()
            accuracy = correct / len(Y)
        return loss.item(), accuracy

    def save_model(self, m_name):
        torch.save(self.model.state_dict(), m_name + '.pth')

    def load_model(self, m_name):
        self.create_model()
        self.model.load_state_dict(torch.load(m_name + '.pth'))


if __name__ == "__main__":
    nb_epoch = 10
    batch_size = 256
    k_folds = 5

    # Mock function to load data
    def get_data(train_size, test_size, num_classes):
        X_train = np.random.rand(train_size, 1, 28, 28)  # Random data, shape [60000, 1, 28, 28]
        Y_train = np.random.randint(0, num_classes, train_size)
        X_test = np.random.rand(test_size, 1, 28, 28)
        Y_test = np.random.randint(0, num_classes, test_size)
        return X_train, Y_train, X_test, Y_test

    X_train, Y_train, X_test, Y_test = get_data(60000, 10000, 10)

    X_folds = np.array_split(X_train, k_folds)
    Y_folds = np.array_split(Y_train, k_folds)

    Model = ModelDM(lr=0.001, img_rows=28, img_cols=28, img_channels=1, n_l1=16, n_l2=None, dout=0, n_filt=8, opt='Adam', regularizer=None, verbose=True, seed=1338)
    Model.create_model()

    k = 1
    X_train_fold = list(X_folds)
    X_test = X_train_fold.pop(k)
    X_train_fold = np.concatenate(X_train_fold)
    Y_train_fold = list(Y_folds)
    Y_test = Y_train_fold.pop(k)
    Y_train_fold = np.concatenate(Y_train_fold)

    Model.train(X_train_fold, Y_train_fold, X_test, Y_test, batch_size, nb_epoch)
    score_reg = Model.evaluate(X_test, Y_test)
    print(f"Test Loss: {score_reg[0]}, Test Accuracy: {score_reg[1]}")

    Model.save_model("a_model")
