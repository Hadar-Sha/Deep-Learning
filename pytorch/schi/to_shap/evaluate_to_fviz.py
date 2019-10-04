"""Evaluates the model"""

import numpy as np
from torch.autograd import Variable
import logging


def evaluate(model, loss_fn, dataloader, metrics, incorrect, epoch):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        incorrect: a function that save all samples with incorrect classification
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        epoch:
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    prop = []
    num_classes = 10
    batch_size = 100
    num_epochs = 10000

    # incorrect samples of current loop
    incorrect_samples = []
    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        if labels_batch.size(1) == 1:
            labels_batch = labels_batch.view(labels_batch.size(0))
        
        # compute model output
        _, _, _, _, output_batch, _ = model(data_batch)
        loss = loss_fn(output_batch, labels_batch, num_classes)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        proportions_batch = labels_batch.shape[0] / batch_size
        prop.append(proportions_batch)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)*proportions_batch
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

        # find incorrect samples
        incorrect_batch = incorrect(data_batch, output_batch, labels_batch)
        incorrect_samples.extend(incorrect_batch)

    # compute mean of all metrics in summary
    prop_sum = np.sum(prop)
    metrics_mean = {metric: np.sum([x[metric] for x in summ]/prop_sum) for metric in summ[0]}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())

    # print to screen every 1% of iterations
    if (epoch+1) % (0.01*num_epochs) == 0:
        print("eval Epoch {}/{}".format(epoch + 1, num_epochs))
        print(metrics_string)
        logging.info("eval Epoch {}/{}".format(epoch + 1, num_epochs))
        logging.info(metrics_string)

    return metrics_mean, incorrect_samples
