import logging
import os
from torch.utils import data
from dataset import Dataset_full, Dataset_IID_window, Dataset_full_SHHS, ConcatDataset, \
    Dataset_IID_window_SHHS, Dataset_Philips_full
import pickle as pkl


def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
    Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
    """

    compare = predictions == targets
    # compare = (predictions.argmax(dim=1)) == (targets)
    # compare = (predictions.argmax(dim=1)) == (targets.argmax(dim=1))
    # summed = compare.sum().item()
    summed = compare.sum()
    # print(summed, compare.size())
    # print(compare.size()[0])
    return summed/compare.size


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_dataloader(data_folder, model_name, data_name, size="default"):
    """
    Returns the correct dataset object given model name and dataset name for training and validation
    """
    training_set = None
    validation_set = None

    if model_name == "Howe_Patterson":
        if data_name == "combined":
            partition = []
            for data_fold in data_folder:
                partition.append(load_obj(os.path.join(data_fold, 'data_partition.pkl')))

        elif size == "small":
            partition = load_obj(os.path.join(data_folder, 'data_partition_small.pkl'))
        elif size == "tiny":
            partition = load_obj(os.path.join(data_folder, 'data_partition_tiny.pkl'))
        else:
            partition = load_obj(os.path.join(data_folder, 'data_partition.pkl'))


        if data_name == "SHHS":
            training_set = Dataset_full_SHHS(partition['train'], data_folder)
            validation_set = Dataset_full_SHHS(partition['validation'], data_folder)
        elif data_name == "snooze":
            training_set = Dataset_full(partition['train'], data_folder)
            validation_set = Dataset_full(partition['validation'], data_folder)
        elif data_name == "philips":
            training_set = Dataset_Philips_full(partition['train'], data_folder)
            validation_set = Dataset_Philips_full(partition['validation'], data_folder)
        elif data_name == "HMC":
            print("{} not implemented data".format(data_name))
            exit()
        elif data_name == "combined":
            training_set = ConcatDataset(
                    Dataset_full(partition[0]['train'], data_folder[0], downsample_ratio=4,
                                  pre_allocation=2 ** 22, down_sample_annotation=False),
                    Dataset_full_SHHS(partition[1]['train'], data_folder[1], downsample_ratio=4,
                                      pre_allocation=2 ** 22, down_sample_annotation=False))
            validation_set = ConcatDataset(
                    Dataset_full(partition[0]['validation'], data_folder[0], downsample_ratio=4,
                                  pre_allocation=2 ** 22, down_sample_annotation=False),
                    Dataset_full_SHHS(partition[1]['validation'], data_folder[1], downsample_ratio=4,
                                      pre_allocation=2 ** 22, down_sample_annotation=False))
        else:
            print("{} wrong data for dataloader".format(data_name))
            exit()
    elif model_name == "Deep_Sleep":
        if data_name == "combined":
            partition = []
            for data_fold in data_folder:
                partition.append(load_obj(os.path.join(data_fold, 'data_partition.pkl')))

        elif size == "small":
            partition = load_obj(os.path.join(data_folder, 'data_partition_small.pkl'))
        elif size == "tiny":
            partition = load_obj(os.path.join(data_folder, 'data_partition_tiny.pkl'))
        else:
            partition = load_obj(os.path.join(data_folder, 'data_partition.pkl'))

        if data_name == "SHHS":
            training_set = Dataset_full_SHHS(partition['train'], data_folder, downsample_ratio=4,
                                             pre_allocation=2 ** 22, down_sample_annotation=False)
            validation_set = Dataset_full_SHHS(partition['validation'], data_folder, downsample_ratio=4,
                                               pre_allocation=2 ** 22, down_sample_annotation=False)
        elif data_name == "snooze":
            training_set = Dataset_full(partition['train'], data_folder, downsample_ratio=4,
                                             pre_allocation=2 ** 22, down_sample_annotation=False)
            validation_set = Dataset_full(partition['validation'], data_folder, downsample_ratio=4,
                                             pre_allocation=2 ** 22, down_sample_annotation=False)
        elif data_name == "philips":
            training_set = Dataset_Philips_full(partition['train'], downsample_ratio=4,
                                             pre_allocation=2 ** 22, down_sample_annotation=False)
            validation_set = Dataset_Philips_full(partition['validation'], downsample_ratio=4,
                                             pre_allocation=2 ** 22, down_sample_annotation=False)
        elif data_name == "HMC":
            print("{} not implemented data".format(data_name))
            exit()
        elif data_name == "combined":
            # TODO combined dataset https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/17
            training_set = ConcatDataset(
                    Dataset_full(partition[0]['train'], data_folder[0], downsample_ratio=4,
                                  pre_allocation=2 ** 22, down_sample_annotation=False),
                    Dataset_full_SHHS(partition[1]['train'], data_folder[1], downsample_ratio=4,
                                      pre_allocation=2 ** 22, down_sample_annotation=False))
            validation_set = ConcatDataset(
                    Dataset_full(partition[0]['validation'], data_folder[0], downsample_ratio=4,
                                  pre_allocation=2 ** 22, down_sample_annotation=False),
                    Dataset_full_SHHS(partition[1]['validation'], data_folder[1], downsample_ratio=4,
                                      pre_allocation=2 ** 22, down_sample_annotation=False))
        else:
            print("{} wrong data for dataloader".format(data_name))
            exit()
    elif model_name == "ConvNet_IID":
        if data_name == "combined":
            partition = []
            for data_fold in data_folder:
                partition.append(load_obj(os.path.join(data_fold, 'data_partition_IID_windows.pkl')))
        else:
            partition = load_obj(os.path.join(data_folder, 'data_partition_IID_windows.pkl'))
        if data_name == "SHHS":
            training_set = Dataset_IID_window_SHHS(partition['train'], data_folder)
            validation_set = Dataset_IID_window_SHHS(partition['validation'], data_folder)
        elif data_name == "snooze":
            training_set = Dataset_IID_window(partition['train'], data_folder)
            validation_set = Dataset_IID_window(partition['validation'], data_folder)
        elif data_name == "philips":
            print("{} not implemented data".format(data_name))
            exit()
        elif data_name == "HMC":
            print("{} not implemented data".format(data_name))
            exit()
        elif data_name == "combined":
            training_set = ConcatDataset(
                Dataset_IID_window(partition[0]['train'], data_folder[0]),
                             Dataset_IID_window_SHHS(partition[1]['train'], data_folder[1]))
            validation_set = ConcatDataset(
                Dataset_IID_window(partition[0]['validation'], data_folder[0]),
                             Dataset_IID_window_SHHS(partition[1]['validation'], data_folder[1]))
        else:
            print("{} wrong data for dataloader".format(data_name))
            exit()

    else:
        print("{} wrong model for dataloader".format(model_name))
        exit()

    return training_set, validation_set