import csv
import os
import numpy as np
import argparse

NUM_OF_CLASSES = 10
NUM_OF_OUTPUT_FILES = 3

parser = argparse.ArgumentParser()
parser.add_argument('--parFolder', default='C:/Users/H/Documents/Haifa Univ/Thesis/deep-learning/deep-input/exp-and-synthetic-with-grayscale/'
                    , help="parent Directory containing input files")

parser.add_argument('--subfolder', default='data/', help="Directory containing input files")
parser.add_argument('--outFolder', default='data-divided-by-100', help="Directory containing output files")
parser.add_argument('--dataSize', default=2429, type=int, help="")
parser.add_argument('--isDevidedByHundred', default=False, help="")


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
        db_reader = csv.reader(csv_file)
        for row in db_reader:
            color = row[0:24]
            # digit = row[-1]
            # digit = row[-2]
            hide_digit = row[label_index]
            confuse_digit = row[-2]
            temp_digit = int(hide_digit)
            if temp_digit == -1:  # ignore samples with "no digit" tag
                continue

            new_row = []
            new_row.extend(color)
            new_row.append(confuse_digit)
            new_row.append(hide_digit)

            # save rows indexes of label = i
            divided_data_indx[temp_digit].append(file_len)
            file_len += 1  # save current file length

            data_list.append(new_row)
    return file_len, data_list, divided_data_indx


def split_data_to_output_files(divided_data_indx, data_list, proportions):
    """
    takes input data file and split to output files according to labels and wanted amounts
    :param divided_data_indx: (list of list) NUM_OF_CLASSES lists. each containing indices of samples with same label
    :param data_list: data from current input file
    :param proportions: how much out of all data needed in each output file

    :return: amount_of_chosen_samples: amount of samples in each output file
    """

    global totalVals
    files_lens = [0] * NUM_OF_OUTPUT_FILES
    splited_data = [[] for _ in range(NUM_OF_OUTPUT_FILES)]  # will contain mat of values for train / dev / test

    for ind1 in range(NUM_OF_CLASSES):  # iterate digits 0-9
        # get list of all samples indices with label = i
        available_indices = divided_data_indx[ind1]
        list_len = len(available_indices)

        for ind2 in range(NUM_OF_OUTPUT_FILES):  # iterate over train/ dev/ test
            # calculates how many samples are needed from label = i to output file = j
            if ind2 < NUM_OF_OUTPUT_FILES-1:
                files_lens[ind2] = int(round(proportions[ind2] * list_len))
            else:
                files_lens[2] = list_len - files_lens[0] - files_lens[1]

            totalVals[ind2] += files_lens[ind2]

            if files_lens[ind2] > 0:
                # choose samples with label = i from available samples and delete chosen samples
                chosen_indices = np.random.choice(available_indices, files_lens[ind2], replace=False)
                # erase chosen samples from available samples list
                available_indices = [item for item in available_indices if item not in chosen_indices]
                # save chosen samples in matrix that will be written to compatible output file
                for elem in chosen_indices:
                    splited_data[ind2].append(data_list[elem])

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

    for ind in range(NUM_OF_OUTPUT_FILES):  # iterate over train/ dev/ test
        os.chdir(currdir)
        os.chdir(output_folder_parent_path)
        output_file_path = os.path.basename(str(output_paths[ind]))
        dirname_to_check = os.path.dirname(str(output_paths[ind]))

        if not os.path.isdir(dirname_to_check):
            os.mkdir(dirname_to_check)
            os.chdir(dirname_to_check)

        else:
            os.chdir(dirname_to_check)

        # write collected data to output file

        with open(output_file_path, 'a+', newline='') as csv_file:
            my_writer = csv.writer(csv_file, delimiter=',')
            for out_row in splited_data[ind]:
                my_writer.writerow(out_row)

    return


def split_extra_to_output(needed_vals, divided_data_indx, proportions, data_list):
    tempNums = [0] * 3
    splited_data = [[] for _ in range(NUM_OF_OUTPUT_FILES)]  # will contain mat of values for train / dev / test

    for i1 in range(NUM_OF_CLASSES):  # for every digit possible
        for j1 in range(NUM_OF_OUTPUT_FILES):  # for train/ dev / test
            current_list = divided_data_indx[i1]
            list_len = len(current_list)

            if i1 < NUM_OF_CLASSES-1:
                # calculate how many samples are needed from a certain digit
                extra_list_len = int(round(proportions[j1] * list_len))
                if tempNums[j1] + extra_list_len < needed_vals[j1]:
                    tempNums[j1] += extra_list_len
                else:
                    extra_list_len = needed_vals[j1] - tempNums[j1]
                    tempNums[j1] = needed_vals[j1]

            else:  # i1==9
                extra_list_len = needed_vals[j1] - tempNums[j1]

            if extra_list_len > 0:
                if extra_list_len < len(current_list):
                    chosen_list = np.random.choice(current_list, extra_list_len, replace=False)
                else:
                    chosen_list = current_list
                    current_list = [item for item in current_list if item not in chosen_list]
                for el in chosen_list:
                    splited_data[j1].append(data_list[el])

    return splited_data


def calc_extra_samples_needed(proportions, data_size, amount_of_chosen_samples):
    """
    :param data_size: total amount of data in all input files
    :param proportions: how much out of all data needed in each output file
    :param amount_of_chosen_samples: how much data is already in every output file
    :return: needed_values_to_add: how much data needs to be added to each output file
    """
    # uncomment in case splitting data from few files(experiment data)
    # and need to add more data from another file named "synthetic"
    needed_values_to_add = [0]*3

    for indx in range(NUM_OF_OUTPUT_FILES):
        # calculate how many missing samples should be taken from the 'extra file'
        needed_values_to_add[indx] = int(proportions[indx] * data_size) - amount_of_chosen_samples[indx]
    return needed_values_to_add


def complete_to_mult(all_vals):
    mult_extra_needed = [0]*3
    for indx in range(NUM_OF_OUTPUT_FILES):
        mult_extra_needed[indx] = round(all_vals[indx], -2) - all_vals[indx]
    return mult_extra_needed


if __name__ == '__main__':

    trainDevTestSizes = [0.7, 0.15, 0.15]  # [0.6, 0.2, 0.2] # [2 / 3, 1 / 6, 1 / 6]
    totalVals = [0] * 3
    lens = [0] * 3
    neededVals = [0] * 3
    allVals = [0] * 3
    dummyProportions = [1] * 3

    # Load the parameters from json file
    args = parser.parse_args()

    folder = args.parFolder + args.subfolder
    outfileLists = [[] for _ in range(NUM_OF_OUTPUT_FILES)]
    syntheticFile = folder + 'synthetic.csv'

    currdir = os.getcwd()
    os.chdir(folder)

    # create paths to output files
    series = 'hidden-combined'
    filesPaths = []
    for it in ['train', 'dev', 'test']:
        path = '{}/'.format(it) + '{}-data-{}.csv'.format(series, it)
        filesPaths.append(path)

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

        infileLen, outfileNew, digitsCountList = read_and_divide_input_to_classes(infilepath, -1)

        splited_data = split_data_to_output_files(digitsCountList, outfileNew, trainDevTestSizes)
        write_splitted_to_files(splited_data, args.outFolder, filesPaths)

    if len(os.listdir(folder)) > 1:
        neededVals = calc_extra_samples_needed(trainDevTestSizes, args.dataSize, totalVals)

        # read extra file and split according to classes
        syntheticFileLen, syntheticFileNew, syntheticDigitsCountList = read_and_divide_input_to_classes(syntheticFile, -1)

        extra_splited_data = split_extra_to_output(neededVals, syntheticDigitsCountList, trainDevTestSizes, syntheticFileNew)
        write_splitted_to_files(extra_splited_data, args.outFolder, filesPaths)

        for i in range(len(totalVals)):
            allVals[i] = totalVals[i] + neededVals[i]

        if args.isDevidedByHundred:
            extraFile = folder + 'synthetic-extra.csv'
            # read extra file and split according to classes
            extraFileLen, extraFileNew, extraDigitsCountList = read_and_divide_input_to_classes(extraFile, -1)

            extraNeededVals = complete_to_mult(allVals)

            second_extra_splited_data = split_extra_to_output(extraNeededVals, extraDigitsCountList, dummyProportions, extraFileNew)
            write_splitted_to_files(second_extra_splited_data, args.outFolder, filesPaths)
