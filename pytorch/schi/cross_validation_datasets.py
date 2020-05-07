import csv
import os
import numpy as np
import argparse
import torch
import pandas as pd
import itertools



NUM_OF_CLASSES = 10
NUM_OF_OUTPUT_FILES = 3
K_FOLD = 6

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--parFolder', default='toSync/Thesis/deep-learning/deep-input/exp-all/exp-and-synthetic-w-gray-total-4000/'
                    , help="parent Directory containing input files") # default='toSync/Thesis/deep-learning/deep-input/dummy-files/'

parser.add_argument('--subfolder', default='data/', help="Directory containing input files")
parser.add_argument('--outFolder', default="data/with-grayscale/k-fold-cross-val ", help="Directory containing output files") # data-dummy
parser.add_argument('--dataSize', default=4000, type=int, help="total size of data") # default=300


def read_and_divide_input_to_classes(file_path, label_index):
    """
    :param file_path: path to file to be split according to classes
    :param label_index: column in which label appears in file
    :return: file_len: amount of samples in the file
            data_list: only relevant data (colors and label) in a list
            divided_data: data divided to NUM_OF_CLASSES lists according to label value
    """

    file_len = 0
    data_list = []
    divided_data_indx = [[] for _ in range(NUM_OF_CLASSES)]
    with open(file_path, newline='') as csv_file:
        db_reader = csv.reader(csv_file)  # , delimiter=' ', quotechar='|'
        for row in db_reader:
            color = row[0:24]
            # digit = row[-1]
            # digit = row[-2]
            digit = row[label_index]
            temp_digit = int(digit)
            if temp_digit == -1:  # ignore samples with "no digit" tag
                continue

            new_row = []
            new_row.extend(color)
            new_row.append(digit)

            # save rows indexes of label = i
            divided_data_indx[temp_digit].append(file_len)
            file_len += 1  # save current file length

            data_list.append(new_row)
    return file_len, data_list, divided_data_indx


def apply_cross_val_per_class(divided_data_indx, data_list, proportions_dict):
    """
    :param divided_data_indx:(list of list) NUM_OF_CLASSES lists. each containing indices of samples with same label
    :param proportions_dict: how much out of all data needed in each output file
    :return: k dictionaries of samples divided to train/ dev/ test
    """
    global totalVals
    splited_data = [{'test': [], 'dev': [], 'train': []} for _ in range(K_FOLD)]  # will contain mat of values for train / dev / test
    # splited_data = [[[] for _ in range(NUM_OF_OUTPUT_FILES)] for _ in range(K_FOLD)]  # will contain mat of values for train / dev / test

    for ind1 in range(NUM_OF_CLASSES):  # iterate digits 0-9
        # get list of all samples indices with label = i
        available_indices = divided_data_indx[ind1]
        list_len = len(available_indices)

        if list_len > 0:
            perm = np.random.permutation(available_indices).tolist()

            ind_dict = [{'test': [], 'dev': [], 'train': []} for _ in range(K_FOLD)]
            for fold_ind in range(K_FOLD):
                cumu_len = 0
                test_len = int(round(proportions_dict['test'] * list_len))  # do not change. constant shift size betweeen folds
                for file_type, value in proportions_dict.items():  # iterate over train/ dev/ test
                    rel_len = int(round(value * list_len))
                    start_ind = (fold_ind*test_len + cumu_len) % list_len
                    end_ind = (fold_ind*test_len + cumu_len + rel_len) % list_len
                    if end_ind < start_ind:
                        end_ind = list_len
                    ind_dict[fold_ind][file_type] = perm[start_ind:end_ind]
                    # if end of list, add from beginning the remaining items to complete to total needed samples
                    if len(ind_dict[fold_ind][file_type]) < rel_len:
                        to_add_len = rel_len - len(ind_dict[fold_ind][file_type])
                        ind_dict[fold_ind][file_type].extend(perm[:to_add_len])

                    cumu_len = cumu_len + rel_len

                # to do: if exists samples that did not add to train / dev/ test (due to rounding) add to test
                is_all_list = []
                for k in ind_dict[fold_ind].keys():
                    is_all_list.extend(ind_dict[fold_ind][k])
                missing_ind = list(set(available_indices) - set(is_all_list))
                ind_dict[fold_ind]['train'].extend(missing_ind)
                # if len(missing_ind) > 0:
                # for item in missing_ind:
                #     ind_dict[fold_ind]['train'].extend(item)
                    # ind_dict[fold_ind]['train'].extend(item)
                # also: if a sample is in more than one file -> remove
                # random_iterator = iter(ind_dict.keys())
                for pair in itertools.combinations(proportions_dict.keys(), 2):
                    intersect_ind = list(set(ind_dict[fold_ind][pair[0]]) & set(ind_dict[fold_ind][pair[1]]))
                    # if len(intersect_ind) > 0:
                    for i in intersect_ind:
                        to_rem = pair[0]
                        if to_rem == 'train':
                            to_rem = pair[1]
                        ind_dict[fold_ind][to_rem].remove(i)

                for file_type, value in proportions_dict.items():  # iterate over train/ dev/ test
                    # retrive data according to chosen indices
                    for elem in ind_dict[fold_ind][file_type]:
                        splited_data[fold_ind][file_type].append(data_list[elem])

                    # to do : continue
                    totalVals[fold_ind][file_type] += len(ind_dict[fold_ind][file_type])
                    # totalVals[fold_ind][file_type] += rel_len

    return splited_data


def write_splitted_to_files(splited_data, output_folder_parent_path, output_paths):
    """
    :param splited_data:
    :param output_folder_parent_path: base folder that will contain splitted output files
    :param output_paths: paths for each output file

    :return:
    """
    os.chdir(currdir)

    if not os.path.isdir(output_folder_parent_path):
        os.chdir(os.path.dirname(output_folder_parent_path))
        os.mkdir(os.path.basename(output_folder_parent_path))
    else:
        os.chdir(output_folder_parent_path)

    for fold_ind in range(K_FOLD):
        for ind in range(NUM_OF_OUTPUT_FILES):  # iterate over train/ dev/ test
            os.chdir(currdir)
            os.chdir(output_folder_parent_path)
            output_file_path = os.path.basename(str(output_paths[fold_ind][ind]))
            dirname_to_check = os.path.dirname(str(output_paths[fold_ind][ind]))
            dir_struct = dirname_to_check.split('/')
            fold_dir = dir_struct[0]
            ftype_dir = dir_struct[1]
            if not os.path.isdir(fold_dir):
                os.mkdir(fold_dir)

            os.chdir(fold_dir)
            # os.chdir(currdir)
            # os.chdir(output_folder_parent_path)
            # output_file_path = os.path.basename(str(output_paths[fold_ind][ind]))
            # dirname_to_check = os.path.dirname(str(output_paths[fold_ind][ind]))

            if not os.path.isdir(ftype_dir):
                os.mkdir(ftype_dir)
                # os.chdir(ftype_dir)

            # else:
            os.chdir(ftype_dir)

            # write collected data to output file
            with open(output_file_path, 'a+', newline='') as csv_file:
                my_writer = csv.writer(csv_file, delimiter=',')
                for out_row in splited_data[fold_ind][ftype_dir]:
                    my_writer.writerow(out_row)
    return


def calc_extra_samples_needed(proportions_dict, data_size, amount_of_chosen_samples):
    """
    :param data_size: total amount of data in all input files
    :param proportions_dict: how much out of all data needed in each output file
    :param amount_of_chosen_samples: how much data is already in every output file
    :return: needed_values_to_add: how much data needs to be added to each output file
    """
    # uncomment in case splitting data from few files(experiment data)
    # and need to add more data from another file named "synthetic"
    needed_values_to_add = [{'test': 0, 'dev': 0, 'train': 0} for _ in range(K_FOLD)]

    for fold_ind in range(K_FOLD):
        for file_type, value in proportions_dict.items():
            # calculate how many missing samples should be taken from the 'extra file'
            needed_values_to_add[fold_ind][file_type] = \
                int(proportions_dict[file_type] * data_size) - amount_of_chosen_samples[fold_ind][file_type]
    return needed_values_to_add


def split_extra_to_output(needed_vals, divided_data_indx, proportions_dict, data_list):
    all_digits_counts = [{'test': 0, 'dev': 0, 'train': 0} for _ in range(K_FOLD)]  # [0] * 3
    splited_data = [{'test': [], 'dev': [], 'train': []} for _ in range(K_FOLD)]  # will contain mat of values for train / dev / test

    for digit in range(NUM_OF_CLASSES):  # for every digit possible
        available_indices = divided_data_indx[digit]
        list_len = len(available_indices)
        perm = np.random.permutation(available_indices).tolist()
        ind_dict = [{'test': [], 'dev': [], 'train': []} for _ in range(K_FOLD)]
        for fold_ind in range(K_FOLD):
            cumu_len = 0
            test_len = int(round(proportions_dict['test'] * list_len))  # do not change. constant shift size betweeen folds
            for file_type, value in proportions_dict.items():  # for train/ dev / test
                if digit < 9:
                    # calculate how many samples are needed from a certain digit
                    extra_list_len = int(round(value * list_len))
                    if all_digits_counts[fold_ind][file_type] + extra_list_len < needed_vals[fold_ind][file_type]:
                        all_digits_counts[fold_ind][file_type] += extra_list_len
                    else:
                        extra_list_len = needed_vals[fold_ind][file_type] - all_digits_counts[fold_ind][file_type]
                        all_digits_counts[fold_ind][file_type] = needed_vals[fold_ind][file_type]

                else:  # digit==9
                    extra_list_len = needed_vals[fold_ind][file_type] - all_digits_counts[fold_ind][file_type]

                if extra_list_len > 0:
                    if extra_list_len < len(available_indices):
                        start_ind = (fold_ind * test_len + cumu_len) % list_len
                        end_ind = (fold_ind * test_len + cumu_len + extra_list_len) % list_len
                        if end_ind < start_ind:
                            end_ind = list_len
                        ind_dict[fold_ind][file_type] = perm[start_ind:end_ind]
                        # if end of list, add from beginning the remaining items to complete to total needed samples
                        if len(ind_dict[fold_ind][file_type]) < extra_list_len:
                            to_add_len = extra_list_len - len(ind_dict[fold_ind][file_type])
                            ind_dict[fold_ind][file_type].extend(perm[:to_add_len])

                        cumu_len = cumu_len + extra_list_len
                        # chosen_list = np.random.choice(available_indices, extra_list_len, replace=False)
                    else:
                        ind_dict[fold_ind][file_type].extend(perm)
                        # chosen_list = available_indices
                        # available_indices = [item for item in available_indices if item not in chosen_list]

            # to do: if exists samples that did not add to train / dev/ test (due to rounding) add to test
            is_all_list = []
            for k in ind_dict[fold_ind].keys():
                is_all_list.extend(ind_dict[fold_ind][k])
            missing_ind = list(set(available_indices) - set(is_all_list))
            ind_dict[fold_ind]['train'].extend(missing_ind)
            all_digits_counts[fold_ind]['train'] += len(missing_ind)
            # also: if a sample is in more than one file -> remove
            for pair in itertools.combinations(proportions_dict.keys(), 2):
                intersect_ind = list(set(ind_dict[fold_ind][pair[0]]) & set(ind_dict[fold_ind][pair[1]]))
                # if len(intersect_ind) > 0:
                for i in intersect_ind:
                    to_rem = pair[0]
                    if to_rem == 'train':
                        to_rem = pair[1]
                    ind_dict[fold_ind][to_rem].remove(i)
                    all_digits_counts[fold_ind][to_rem] -= 1

            for file_type, value in proportions_dict.items():  # iterate over train/ dev/ test
                for elem in ind_dict[fold_ind][file_type]:
                    splited_data[fold_ind][file_type].append(data_list[elem])

    return splited_data


if __name__ == '__main__':
    proportions_dict = {'train': 0.7, 'dev': 0.15, 'test': 0.15}
    totalVals = [{'test': 0, 'dev': 0, 'train': 0} for _ in range(K_FOLD)]
    lens = [0] * 3
    neededVals = [{'test': 0, 'dev': 0, 'train': 0} for _ in range(K_FOLD)]  # [0] * 3
    allVals = [0] * 3
    dummyProportions = [1] * 3

    # Load the parameters from json file
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        # os.chdir(os.path.join(os.path.expanduser(os.environ['OneDrive']), args.parent_dir))
        args.parent_dir = os.path.join(os.path.expanduser(os.environ['OneDrive']), args.parent_dir)
        args.parFolder = os.path.join(os.path.expanduser(os.environ['OneDrive']), args.parFolder)

    folder = os.path.join(os.path.expanduser(os.environ['OneDrive']), args.parFolder, args.subfolder)
    outfileLists = [[] for _ in range(NUM_OF_OUTPUT_FILES)]
    syntheticFile = folder + 'synthetic.csv'

    currdir = os.getcwd()
    os.chdir(folder)

    # create paths to output files
    series = 'synthetic-gray'  # 'synthetic-BW'  # 'exp-synthetic'
    filesPaths = [[] for _ in range(K_FOLD)]
    for fold_i in range(K_FOLD):
        for it in proportions_dict.keys():
            # for it in ['train', 'dev', 'test']:
            path = '{}-fold/'.format(fold_i+1) + '{}/'.format(it) + '{}-data-{}.csv'.format(series, it)
            filesPaths[fold_i].append(path)

    # iterate over input files
    for file in os.listdir(folder):
        filename = os.path.splitext(file)[0]
        # ignore irrelevant files
        if filename == 'synthetic' or filename == 'synthetic-extra':
            continue
        if file.endswith('txt'):
            continue

        infilepath = folder + file
        outFiles = [[] for _ in range(NUM_OF_OUTPUT_FILES)]  # will contain mat of values for train / dev / test

        infileLen, outfileNew, digitsCountList = read_and_divide_input_to_classes(infilepath, -2)  # -1)
        print(infileLen)

        splited_data = apply_cross_val_per_class(digitsCountList, outfileNew, proportions_dict)
        if args.parent_dir and not torch.cuda.is_available():
            args.outFolder = os.path.join(args.parent_dir, args.outFolder)
        write_splitted_to_files(splited_data, args.outFolder, filesPaths)

    if len(os.listdir(folder)) > 1:
        neededVals = calc_extra_samples_needed(proportions_dict, args.dataSize, totalVals)
        # read extra file and split according to classes
        syntheticFileLen, syntheticFileNew, syntheticDigitsCountList = read_and_divide_input_to_classes(syntheticFile, -1)

        extra_splited_data = split_extra_to_output(neededVals, syntheticDigitsCountList, proportions_dict,
                                                   syntheticFileNew)
        # if args.parent_dir and not torch.cuda.is_available():
        #     args.outFolder = args.parent_dir + args.outFolder
        write_splitted_to_files(extra_splited_data, args.outFolder, filesPaths)