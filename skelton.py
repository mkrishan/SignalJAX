import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import copy
import logging
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def recursive_dict():
    return defaultdict(recursive_dict)

def subtract_weights(X_1, X_2):
    """
    Subtracts X_1 from X_2
    :param X_1: list of weights (PyTorch tensors)
    :param X_2: list of weights (PyTorch tensors)
    :return: list of weights (PyTorch tensors)
    """
    return average_weights(X_1, X_2, w1=1.0, w2=-1.0)

def add_weights(X_1, X_2):
    """
    Adds X_1 to X_2
    :param X_1: list of weights (PyTorch tensors)
    :param X_2: list of weights (PyTorch tensors)
    :return: list of weights (PyTorch tensors)
    """
    return average_weights(X_1, X_2, w1=1.0, w2=1.0)

def multiply_weights(X_1, mult):
    """
    Multiplies X_1 by scalar mult
    :param X_1: list of weights (PyTorch tensors)
    :param mult: scalar to multiply weights by
    :return: list of weights (PyTorch tensors)
    """
    return average_weights(X_1, X_1, w1=mult, w2=0)

def add_X(X_1, X_2):
    """
    Adds the dict of weights X_1 to X_2
    :param X_1: dict of weights where each key contains a list of weights (PyTorch tensors)
    :param X_2: dict of weights where each key contains a list of weights (PyTorch tensors)
    :return: dict of weights
    """
    return {ind: add_weights(copy.deepcopy(X_1[ind]), copy.deepcopy(X_2[ind])) for ind in X_1.keys()}

def subtract_X(X_1, X_2):
    """
    Subtracts the dict of weights X_2 from X_1
    :param X_1: dict of weights where each key contains a list of weights (PyTorch tensors)
    :param X_2: dict of weights where each key contains a list of weights (PyTorch tensors)
    :return: dict of weights
    """
    return {ind: subtract_weights(copy.deepcopy(X_1[ind]), copy.deepcopy(X_2[ind])) for ind in X_1.keys()}

def multiply_X(X, mult):
    """
    Scalar multiplication of the dict of weights X
    :param X: dict of weights where each key contains a list of weights (PyTorch tensors)
    :param mult: scalar to multiply weights
    :return: dict of weights
    """
    return {ind: multiply_weights(copy.deepcopy(X[ind]), mult) for ind in X.keys()}

def average_weights(weights_1, weights_2, w1=0.5, w2=0.5):
    """
    Combine two weights using addition/subtraction
    :param weights_1: list of weights (PyTorch tensors)
    :param weights_2: list of weights (PyTorch tensors)
    :param w1: the scalar value to apply to weights_1
    :param w2: the scalar value to apply to weights_2
    :return: the weighted sum of weights_1 and weights_2
    """
    return [w1 * w1_ + w2 * w2_ for w1_, w2_ in zip(weights_1, weights_2)]

def weights_error(weights_1, weights_2):
    """
    Get the L2 norm of weights_1 - weights_2
    :param weights_1: list of weights (PyTorch tensors)
    :param weights_2: list of weights (PyTorch tensors)
    :return: scalar error
    """
    weights = subtract_weights(weights_1, weights_2)
    return sum((w1 - w2).norm().item() for w1, w2 in zip(weights_1, weights_2))

def create_empty_weights(weights_in):
    """
    Create a set of zero weights with same size as weights_in
    :param weights_in: list of weights (PyTorch tensors)
    :return: list of zero weights (PyTorch tensors)
    """
    return [torch.zeros_like(w) for w in weights_in]

def get_data(n_train, n_test, nb_classes):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train, y_train = train_dataset.data[:n_train], train_dataset.targets[:n_train]
    X_test, y_test = test_dataset.data[:n_test], test_dataset.targets[:n_test]

    # Reshape and normalize
    X_train = X_train.unsqueeze(1).float() / 255.0  # (n_train, 1, 28, 28)
    X_test = X_test.unsqueeze(1).float() / 255.0  # (n_test, 1, 28, 28)

    print(f'{X_train.shape[0]} train samples')
    print(f'{X_test.shape[0]} test samples')

    # Convert class vectors to one-hot encoded matrices
    Y_train = torch.eye(nb_classes)[y_train]
    Y_test = torch.eye(nb_classes)[y_test]

    return X_train, Y_train, X_test, Y_test

def get_cifar(nb_classes=10):
    # Load CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    X_train, y_train = train_dataset.data, train_dataset.targets
    X_test, y_test = test_dataset.data, test_dataset.targets

    X_train = torch.tensor(X_train).permute(0, 3, 1, 2).float() / 255.0  # Convert to (n_train, 3, 32, 32)
    X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float() / 255.0  # Convert to (n_test, 3, 32, 32)

    print(f'X_train shape: {X_train.shape}')
    print(f'{X_train.shape[0]} train samples')
    print(f'{X_test.shape[0]} test samples')

    # Convert class vectors to one-hot encoded matrices
    Y_train = torch.eye(nb_classes)[torch.tensor(y_train)]
    Y_test = torch.eye(nb_classes)[torch.tensor(y_test)]

    return X_train, Y_train, X_test, Y_test

def add_weights_mask(weights_1, weights_2, mask):
    # weights_1: original, weights_2: other
    return [w1 * m + (1 - m) * w2 for w1, w2, m in zip(weights_1, weights_2, mask)]

def create_mask(weights, p=0.5):
    return [(torch.rand_like(layer) > p).int() for layer in weights]

def soft_threshold_weights(weights_in, thresh):
    """
    L1 soft-threshold operation
    """
    weights_out = [torch.zeros_like(w) for w in weights_in]
    for i, weight in enumerate(weights_in):
        w = weight.clone()
        w[torch.abs(w) < thresh] = 0
        w[w >= thresh] -= thresh
        w[w <= -thresh] += thresh
        weights_out[i] = w
    return weights_out

def dm_train(model, X_train, Y_train, X=None, X_test=None, Y_test=None, n_set=2, rand_starts=False, switch_projections=True,
             iterations=20, val_lim=0.99, nb_epoch=1200, use_dm=True, seed=1338, model_func=None,
             init_weights=None, batch_size=None, verbose=True, return_average=False, average_start=50, beta=1.0,
             thresh=None, reset_momentum=False, early_term=True):
    """
    Training weights using the difference map
    """
    if verbose:
        print(n_set, iterations, val_lim, nb_epoch, batch_size)

    algo = {True: 'DM', False: 'ER'}

    # Split data into constraint sets
    X_trains = {ind: torch.split(X_train, int(X_train.shape[0] / n_set), dim=0)[ind] for ind in range(n_set)}
    Y_trains = {ind: torch.split(Y_train, int(Y_train.shape[0] / n_set), dim=0)[ind] for ind in range(n_set)}

    scores = []
    scores_train = []
    dm_errors = []

    if seed is not None:
        torch.manual_seed(seed)

    if X is None:
        if rand_starts:
            X = {ind: model_func().state_dict() for ind in range(n_set)}
        else:
            if init_weights is None:
                raise ValueError("init_weights not specified.")
            X = {ind: copy.deepcopy(init_weights) for ind in range(n_set)}
    else:
        X = {ind: copy.deepcopy(init_weights) for ind in range(n_set)}

    X_A = P_data(X, X_trains, Y_trains, model, nb_epoch, val_lim, batch_size)
    if switch_projections:
        X_A = P_data(X, X_trains, Y_trains, model, nb_epoch, val_lim, batch_size)
    else:
        X_A = P_avg(X, thresh)

    for ii in range(iterations):
        t0 = time.time()

        if use_dm:
            X_R = subtract_X(multiply_X(X_A, 2.0), X)
            if switch_projections:
                X_P = P_avg(X_R, thresh)
            else:
                X_P = P_data(X_R, X_trains, Y_trains, model, nb_epoch, val_lim, batch_size)

            X_D = subtract_X(X_P, X_A)
            X = add_X(X, multiply_X(X_D, beta))

            dm_error = weights_error(X_A[0], P_avg(X, thresh)[0])
            dm_errors.append(dm_error)

            if switch_projections:
                X_A = P_data(X, X_trains, Y_trains, model, nb_epoch, val_lim, batch_size)
            else:
                X_A = P_avg(X, thresh)

            if return_average and ii > average_start:
                average_converged_weights.append(P_avg(X, thresh)[0])

        else:
            X = P_data(X_A, X_trains, Y_trains, model, nb_epoch, val_lim, batch_size)

            dm_error = weights_error(X_A[0], P_avg(X, thresh)[0])
            dm_errors.append(dm_error)

            X_A = P_avg(X, thresh)

        model.load_state_dict(P_avg(X, thresh)[0])
        score = model.evaluate(X_test, Y_test)
        scores.append(score)
        scores_train.append(model.evaluate(X_train, Y_train))

        if verbose:
            print(n_set, score, ii, scores_train[-1], dm_error, algo[use_dm], time.time() - t0)

    weights_out = P_avg(X, thresh)[0]
    if return_average and len(average_converged_weights) != 0:
        weights_out = average_all_weights(average_converged_weights)

    model.load_state_dict(weights_out)
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return model, weights_out, [s[0] for s in scores], [s[1] for s in scores], [s[0] for s in scores_train], [s[1] for s in scores_train], dm_errors
