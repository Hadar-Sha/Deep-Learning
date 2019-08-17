"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable


class DiscriminatorNet(nn.Module):
    def __init__(self, params):
        super(DiscriminatorNet, self).__init__()

        self.in_layer = nn.Sequential(
            nn.Linear(params.input_size, params.hidden_size),
            nn.LeakyReLU(params.leaky_relu_slope)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.LeakyReLU(params.leaky_relu_slope)
        )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(params.hidden_size, params.hidden_size),
        #     nn.LeakyReLU(params.leaky_relu_slope)
        # )

        self.out_layer_real_fake = nn.Sequential(
            nn.Linear(params.hidden_size, 1),
            nn.Sigmoid()
        )

        # to output 2 labels we output a vector of size 2 * num classes
        self.out_layer_class = nn.Sequential(
            nn.Linear(params.hidden_size, 2*params.num_classes),
            nn.LogSoftmax(dim=1)    # remove !!!!!!!!!!!!!
            # nn.Softmax(dim=1)    # remove !!!!!!!!!!!!!

        )

    def forward(self, x):

        x_ = self.in_layer(x)
        x_ = self.hidden1(x_)
        # x_ = self.hidden2(x_)
        # x_ = self.hidden3(x_)

        out_real_fake = self.out_layer_real_fake(x_)
        out_class = self.out_layer_class(x_)

        return out_real_fake, out_class


class GeneratorNet(nn.Module):
    def __init__(self, params):
        super(GeneratorNet, self).__init__()

        self.hidden_with_label = nn.Sequential(
            nn.Linear(params.noise_dim + 2*params.num_classes, params.hidden_size),
            nn.LeakyReLU(params.leaky_relu_slope)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.LeakyReLU(params.leaky_relu_slope)
        )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(params.hidden_size, params.hidden_size),
        #     nn.LeakyReLU(params.leaky_relu_slope)
        # )
        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size, params.input_size),
            nn.Tanh()
        )

    def forward(self, x, labels):

        out = torch.cat([labels, x], 1)
        # out = torch.cat([x, labels], 1)

        out = self.hidden_with_label(out)
        out = self.hidden1(out)
        # out = self.hidden2(out)
        # out = self.hidden3(out)
        out = self.out_layer(out)

        return out


# Noise
def noise(size, dim, noise_type='normal'):
    if noise_type == 'normal':
        n = Variable(torch.randn(size, dim))  # recommended to sample from normal distribution and not from uniform dist
    elif noise_type == 'uniform':
        n = Variable(-1 + 2 * torch.rand(size, dim))  # make sense for binary samples
    elif noise_type == 'binary':
        n = Variable(-1 + 2 * torch.randint(2, (size, dim), dtype=torch.float))  # make sense for binary samples
    if torch.cuda.is_available():
        return n.cuda()
    return n


def convert_int_to_one_hot_vector(label, num_of_classes):

    if len(list(label.size())) < 3:
        label_shaped = label.view(-1, 1)

        one_hot_vector = torch.zeros([list(label.size())[0], num_of_classes], device=label.device)

        one_hot_vector.scatter_(1, label_shaped, 1)
        one_hot_vector = one_hot_vector.type(torch.FloatTensor)

        if torch.cuda.is_available():
            return one_hot_vector.cuda()
        return one_hot_vector

    else:
        # this is for 3d tensor
        labels_shaped = label.view(label.size(0), label.size(1), -1)

        one_hot_matrix = torch.zeros([list(labels_shaped.size())[0], list(labels_shaped.size())[1], num_of_classes], device=label.device)
        one_hot_matrix.scatter_(2, labels_shaped, 1)
        # added to keep a 2d dimension of labels
        one_hot_matrix = one_hot_matrix.view(-1, list(labels_shaped.size())[1]*num_of_classes)
        one_hot_matrix = one_hot_matrix.type(torch.FloatTensor)

        if torch.cuda.is_available():
            return one_hot_matrix.cuda()
        return one_hot_matrix


def real_data_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def fake_data_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def vectors_to_samples(vectors):
    vectors = vectors.reshape(vectors.size()[0], -1, 3)
    vectors = vectors.cpu().numpy()
    vectors = vectors.tolist()
    return vectors


def labels_to_titles(labels):
    if len(labels.shape) > 1 and min(labels.shape) == 1:
        labels = labels.view(labels.size()[0],)
    labels = (labels.cpu().numpy()).tolist()
    return labels


# def create_random_labels(num_samples, num_classes, mode="One Label"):
#
#     if mode == "One Label":
#         test_labels = list(range(num_samples))
#         test_labels = [it % num_classes for it in test_labels]
#         test_labels = torch.Tensor(test_labels)
#         test_labels = test_labels.view(num_samples, -1)
#         test_labels = test_labels.type(torch.LongTensor)
#         return test_labels
#
#     if mode == "Two Labels":
#         all_but_eight = np.concatenate((np.arange(8), np.array([[9]])), axis=None)
#         random_mat = np.zeros([num_samples, 2])
#         for i in range(num_samples):
#             random_mat[i][:] = np.random.choice(all_but_eight, 2, replace=False)
#
#         random_tensor = torch.from_numpy(random_mat)
#         random_tensor = random_tensor.type(torch.LongTensor)
#         return random_tensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        # nn.init.uniform_(m.bias.data, -1, 1)

    if torch.cuda.is_available():
        for pa in m.parameters():
            pa.cuda()


def real_fake_loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 10 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
        num_of_classes: (int) value describing number of different classes (10)

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    binary_criterion = nn.BCELoss()
    if torch.cuda.is_available():
        binary_criterion = nn.BCELoss().cuda()

    return binary_criterion(outputs, labels)


# def class_selection_loss_fn(outputs, labels):
#
#     cross_entropy_criterion = nn.CrossEntropyLoss()
#     if torch.cuda.is_available():
#         cross_entropy_criterion = nn.CrossEntropyLoss().cuda()
#
#     return cross_entropy_criterion(outputs, labels)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        # val = - F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        # val = -x*torch.log(x)
        val = -x*torch.exp(x)
        val = torch.sum(val, dim=1)
        # val = -1.0 * val.sum()
        return torch.mean(val)


def class_selection_loss_fn(outputs, labels, num_of_classes):
    """
        Compute the loss given outputs and labels.
        we will achieve max KL dist between out_bef_filt and lab_aft_filt by the following:
        we wish the output to be not equal to lab_aft_filt.
        given lab_aft_filt in [0-9] we can create a binary vector of size [num_of_classes] with all entrances = 1 if
        entrance != lab_aft_filt and entrance = 0 otherwise.
        we normalize this vector v to have sum = 1 by dividing in (num_of_classes-1)
        we then calculate KL loss between out_bef_filt and v

        Args:
            outputs: (Variable) dimension batch_size x 10 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
            num_of_classes: (int) value describing number of different classes (10)

        Returns:
            loss (Variable): loss for all samples in the batch
    """

    # kl_criterion = nn.KLDivLoss(size_average=True, reduce=True)
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    min_entropy_criterion = HLoss()

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0], device=labels.device))
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1], device=labels.device))

    one_hot_vector_after_filter = convert_int_to_one_hot_vector(label_after_filter, num_of_classes)
    one_hot_vector_before_filter = convert_int_to_one_hot_vector(label_before_filter, num_of_classes)  # unneeded

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10)), device=outputs.device))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10, 20)), device=outputs.device))

    completing_after_filter = (torch.ones(labels.shape[0], num_of_classes, device=labels.device) - one_hot_vector_after_filter)\
                               / (num_of_classes-1)

    func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
           kl_criterion(out_before_filter, completing_after_filter) + \
           min_entropy_criterion(out_before_filter)

    return func


def class_selection_loss_fn_exact_equal(outputs, labels, num_of_classes):
    """
        Compute the loss given outputs and labels.
        we will achieve max KL dist between out_bef_filt and lab_aft_filt by the following:
        we wish the output to be not equal to lab_aft_filt.
        given lab_aft_filt in [0-9] we can create a binary vector of size [num_of_classes] with all entrances = 1 if
        entrance != lab_aft_filt and entrance = 0 otherwise.
        we normalize this vector v to have sum = 1 by dividing in (num_of_classes-1)
        we then calculate KL loss between out_bef_filt and v

        Args:
            outputs: (Variable) dimension batch_size x 10 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
            num_of_classes: (int) value describing number of different classes (10)

        Returns:
            loss (Variable): loss for all samples in the batch
    """

    cross_entropy_criterion = nn.CrossEntropyLoss()
    # # kl_criterion = nn.KLDivLoss(size_average=True, reduce=True)
    # kl_criterion = nn.KLDivLoss(reduction='batchmean')
    # min_entropy_criterion = HLoss()

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0], device=labels.device))
    label_before_filter = label_before_filter.view(label_before_filter.shape[0])
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1], device=labels.device))
    label_after_filter = label_after_filter.view(label_after_filter.shape[0])

    # one_hot_vector_after_filter = convert_int_to_one_hot_vector(label_after_filter, num_of_classes)
    # one_hot_vector_before_filter = convert_int_to_one_hot_vector(label_before_filter, num_of_classes)  # unneeded

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10)), device=outputs.device))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10, 20)), device=outputs.device))

    # completing_after_filter = (torch.ones(labels.shape[0], num_of_classes, device=labels.device) - one_hot_vector_after_filter)\
    #                            / (num_of_classes-1)

    func = cross_entropy_criterion(out_before_filter, label_before_filter) + \
           cross_entropy_criterion(out_after_filter, label_after_filter)
    # func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
    #        kl_criterion(out_before_filter, completing_after_filter) + \
    #        min_entropy_criterion(out_before_filter)

    return func


def class_selection_loss_fn_exp_kl(outputs, labels, num_of_classes):
    """
        Compute the loss given outputs and labels.
        we will achieve max KL dist between out_bef_filt and lab_aft_filt by the following:
        we wish the output to be not equal to lab_aft_filt.
        given lab_aft_filt in [0-9] we can create a binary vector of size [num_of_classes] with all entrances = 1 if
        entrance != lab_aft_filt and entrance = 0 otherwise.
        we normalize this vector v to have sum = 1 by dividing in (num_of_classes-1)
        we then calculate KL loss between out_bef_filt and v

        Args:
            outputs: (Variable) dimension batch_size x 10 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
            num_of_classes: (int) value describing number of different classes (10)

        Returns:
            loss (Variable): loss for all samples in the batch
    """

    # kl_criterion = nn.KLDivLoss(size_average=True, reduce=True)
    kl_criterion = nn.KLDivLoss()
    min_entropy_criterion = HLoss()

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0], device=labels.device))
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1], device=labels.device))

    one_hot_vector_after_filter = convert_int_to_one_hot_vector(label_after_filter, num_of_classes)
    one_hot_vector_before_filter = convert_int_to_one_hot_vector(label_before_filter, num_of_classes)  # unneeded

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10)), device=outputs.device))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10, 20)), device=outputs.device))

    completing_after_filter = (torch.ones(labels.shape[0], num_of_classes, device=labels.device) - one_hot_vector_after_filter)\
                               / (num_of_classes-1)

    func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
           (1 - torch.exp(- kl_criterion(out_before_filter, one_hot_vector_after_filter))) + \
           min_entropy_criterion(out_after_filter)
           # min_entropy_criterion(out_before_filter)

    # kl_criterion(out_before_filter, completing_after_filter) + \

    return func


# compute the current classification accuracy
def compute_acc(outputs, labels):
    # outputs_ = outputs.data.max(1)[1]
    # correct = outputs_.eq(labels.data).cpu().sum()
    # acc = float(correct) / float(len(labels.data))  # * 100.0
    # return acc

    if len(list(labels.shape)) == 1:
        return 0.0

    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    label_before_filter = labels[:, 0]  # unneeded
    label_after_filter = labels[:, 1]

    out_before_filter = outputs[:, :10]
    out_after_filter = outputs[:, 10:]

    out_int_before = np.argmax(out_before_filter, axis=1)
    out_int_after = np.argmax(out_after_filter, axis=1)

    # the classification before filter is correct as long as it is not equal to label after filter
    before_ind = np.nonzero(out_int_before != label_after_filter)
    after_ind = np.nonzero(out_int_after == label_after_filter)
    all_ind = np.intersect1d(before_ind, after_ind)

    all_count = all_ind.shape[0]
    return all_count / float(labels.shape[0])


# compute the current classification accuracy
def compute_acc_exact_equal(outputs, labels):
    # outputs_ = outputs.data.max(1)[1]
    # correct = outputs_.eq(labels.data).cpu().sum()
    # acc = float(correct) / float(len(labels.data))  # * 100.0
    # return acc

    if len(list(labels.shape)) == 1:
        return 0.0

    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    label_before_filter = labels[:, 0]  # unneeded
    label_after_filter = labels[:, 1]

    out_before_filter = outputs[:, :10]
    out_after_filter = outputs[:, 10:]

    out_int_before = np.argmax(out_before_filter, axis=1)
    out_int_after = np.argmax(out_after_filter, axis=1)

    # the classification before filter is correct as long as it is not equal to label after filter
    before_ind = np.nonzero(out_int_before == label_before_filter)
    after_ind = np.nonzero(out_int_after == label_after_filter)
    all_ind = np.intersect1d(before_ind, after_ind)

    all_count = all_ind.shape[0]
    return all_count / float(labels.shape[0])


def incorrect(images, outputs, labels, curr_min=-1, curr_max=1, dest_min=0, dest_max=255):
    """
        Keep all images for which the classification is wrong

        Args:
            images: (np.ndarray) dimension batch_size x 24- input to the model
            outputs: (np.ndarray) dimension batch_size x 10 - log softmax output of the model
            labels: (np.ndarray) dimension batch_size, where each element is a value in [0- 9]

        Returns: (list) of images for which the classification is wrong, the classification and the correct label
        """
    mat_out = []

    # images = images.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    label_before_filter = labels[:, 0]  # unneeded
    label_after_filter = labels[:, 1]

    out_before_filter = outputs[:, :10]
    out_after_filter = outputs[:, 10:]

    out_int_before = np.argmax(out_before_filter, axis=1)
    out_int_after = np.argmax(out_after_filter, axis=1)

    # find incorrect indexes
    # the classification before filter is incorrect only if it is equal to label after filter
    correct_before_indexes = np.nonzero(out_int_before != label_after_filter)  # label_after_filter : not a typo!!!!
    correct_after_indexes = np.nonzero(out_int_after == label_after_filter)
    all_indexes = np.arange(out_int_before.shape[0])  # to get all indices
    all_correct_indexes = np.intersect1d(correct_before_indexes, correct_after_indexes)
    incorrect_indexes = np.setdiff1d(all_indexes, all_correct_indexes)

    # find compatible incorrect samples and save them in a list
    samples_numpy = images.cpu().numpy()
    # convert back to range [0, 255]
    samples_numpy = \
        dest_min + (dest_max - dest_min) * (samples_numpy - curr_min) / (curr_max - curr_min)
    samples_numpy = np.around(samples_numpy).astype(int)

    # find samples
    incorrect_samples = (samples_numpy[incorrect_indexes])  # .astype(int)

    # find classifier result for before filter
    incorrect_before_labels = out_int_before[incorrect_indexes]

    # find classifier result for after filter
    incorrect_after_labels = out_int_after[incorrect_indexes]

    # find true labels for before filter
    true_before_labels = label_before_filter[incorrect_indexes]

    # find true labels for after filter
    true_after_labels = label_after_filter[incorrect_indexes]

    # organize data
    before_labels = np.column_stack((incorrect_before_labels, true_before_labels))
    after_labels = np.column_stack((incorrect_after_labels, true_after_labels))
    all_labels = np.column_stack((before_labels, after_labels))

    numpy_mat_out = np.concatenate((incorrect_samples, all_labels), axis=1)
    length = len(numpy_mat_out.tolist())
    if length > 0:
        mat_out.extend(numpy_mat_out.tolist())

    return mat_out


def incorrect_exact_equal(images, outputs, labels, curr_min=-1, curr_max=1, dest_min=0, dest_max=255):
    """
        Keep all images for which the classification is wrong

        Args:
            images: (np.ndarray) dimension batch_size x 24- input to the model
            outputs: (np.ndarray) dimension batch_size x 10 - log softmax output of the model
            labels: (np.ndarray) dimension batch_size, where each element is a value in [0- 9]

        Returns: (list) of images for which the classification is wrong, the classification and the correct label
        """
    mat_out = []

    # images = images.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    label_before_filter = labels[:, 0]  # unneeded
    label_after_filter = labels[:, 1]

    out_before_filter = outputs[:, :10]
    out_after_filter = outputs[:, 10:]

    out_int_before = np.argmax(out_before_filter, axis=1)
    out_int_after = np.argmax(out_after_filter, axis=1)

    # find incorrect indexes
    # the classification before filter is incorrect only if it is equal to label after filter
    correct_before_indexes = np.nonzero(out_int_before == label_before_filter)
    correct_after_indexes = np.nonzero(out_int_after == label_after_filter)
    all_indexes = np.arange(out_int_before.shape[0])  # to get all indices
    all_correct_indexes = np.intersect1d(correct_before_indexes, correct_after_indexes)
    incorrect_indexes = np.setdiff1d(all_indexes, all_correct_indexes)

    # find compatible incorrect samples and save them in a list
    samples_numpy = images.cpu().numpy()
    # convert back to range [0, 255]
    samples_numpy = \
        dest_min + (dest_max - dest_min) * (samples_numpy - curr_min) / (curr_max - curr_min)
    samples_numpy = np.around(samples_numpy).astype(int)

    # find samples
    incorrect_samples = (samples_numpy[incorrect_indexes])  # .astype(int)

    # find classifier result for before filter
    incorrect_before_labels = out_int_before[incorrect_indexes]

    # find classifier result for after filter
    incorrect_after_labels = out_int_after[incorrect_indexes]

    # find true labels for before filter
    true_before_labels = label_before_filter[incorrect_indexes]

    # find true labels for after filter
    true_after_labels = label_after_filter[incorrect_indexes]

    # organize data
    before_labels = np.column_stack((incorrect_before_labels, true_before_labels))
    after_labels = np.column_stack((incorrect_after_labels, true_after_labels))
    all_labels = np.column_stack((before_labels, after_labels))

    numpy_mat_out = np.concatenate((incorrect_samples, all_labels), axis=1)
    length = len(numpy_mat_out.tolist())
    if length > 0:
        mat_out.extend(numpy_mat_out.tolist())

    return mat_out
