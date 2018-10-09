import csv
import os
import numpy as np

trainDevTestSizes = [0.7, 0.15, 0.15]  # [0.6, 0.2, 0.2]  #
NUM_OF_CLASSES = 10
NUM_OF_OUTPUT_FILES = 3


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


def split_data_to_output_files(divided_data_indx, data_list, proportions, output_folder_parent_path, output_paths):
    """
    takes input data file and split to output files according to labels and wanted amounts
    :param divided_data_indx: (list of list) NUM_OF_CLASSES lists. each containing indices of samples with same label
    :param data_list: data from current input file
    :param proportions: how much out of all data needed in each output file
    :param output_folder_parent_path: base folder that will contain splitted output files
    :param output_paths: paths for each output file
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

    for ind3 in range(NUM_OF_OUTPUT_FILES):  # iterate over train/ dev/ test
        os.chdir(currdir)
        output_file_path = output_folder_parent_path + str(output_paths[ind3])

        # write collected data to output file
        with open(output_file_path, 'a+', newline='') as csv_file:
            my_writer = csv.writer(csv_file, delimiter=',')
            for out_row in splited_data[ind3]:
                my_writer.writerow(out_row)
    return


def split_extra_to_output(needed_vals, divided_data_indx, proportions, data_list, output_folder_parent_path, output_paths):
    tempNums = [0] * 3
    splited_data = [[] for _ in range(NUM_OF_OUTPUT_FILES)]  # will contain mat of values for train / dev / test

    for i1 in range(NUM_OF_CLASSES):  # for every digit possible
        for j1 in range(NUM_OF_OUTPUT_FILES):  # for train/ dev / test
            current_list = divided_data_indx[i1]
            list_len = len(current_list)

            if i1 < 9:
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

    for j2 in range(NUM_OF_OUTPUT_FILES):  # iterate over train/ dev/ test
        os.chdir(currdir)
        out_file_path = output_folder_parent_path + str(output_paths[j2])

        print('j: {} len: {}'.format(j2, len(splited_data[j2])))

        with open(out_file_path, 'a+', newline='') as csv_file:
            my_writer = csv.writer(csv_file, delimiter=',')
            for out_row in splited_data[j2]:
                my_writer.writerow(out_row)
    return


# def add_last_samples(needed_vals, divided_data_indx):
#     for i1 in range(NUM_OF_CLASSES):  # for every digit possible
#         for j1 in range(NUM_OF_OUTPUT_FILES):  # for train/ dev / test
#             current_list = divided_data_indx[i1]
#             list_len = len(current_list)
#
#
#     return


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


def complete_to_mult(allVals):
    mult_extra_needed = [0]*3
    for indx in range(NUM_OF_OUTPUT_FILES):
        mult_extra_needed[indx] = round(allVals[indx], -2) - allVals[indx]
    return mult_extra_needed


parFolder = 'C:/Users/H/Documents/Haifa Univ/Thesis/deep-learning/deep-input/exp-and-synthetic-with-grayscale/'
# hidden/gray/'
# exp-and-synthetic-with-grayscale/'#-total-2429/'

folder = parFolder + 'data/'
outFolder = 'data-divided-by-100'  # 'data-hidden-gray'#'data-with-grayscale/' #'./data'

currdir = os.getcwd()
os.chdir(folder)


dataSize = 2429 #2500 #2200
totalVals = [0]*3
lens = [0]*3
neededVals = [0]*3

# create paths to output files
series = 'exp-synthetic'
filesPaths = []
for it in ['train', 'dev', 'test']:
    path = '/{}/'.format(it) + '{}-data-{}.csv'.format(series, it)
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

    infileLen, outfileNew, digitsCountList = read_and_divide_input_to_classes(infilepath, -2)

    split_data_to_output_files(digitsCountList, outfileNew, trainDevTestSizes, outFolder, filesPaths)


neededVals = calc_extra_samples_needed(trainDevTestSizes, dataSize, totalVals)

# outfileNew = []
# extraFileLen = 0
# digitsCountList = [[] for k1 in range(10)]
outfileLists = [[] for _ in range(NUM_OF_OUTPUT_FILES)]

syntheticFile = folder + 'synthetic.csv'

# read extra file and split according to classes
syntheticFileLen, syntheticFileNew, syntheticDigitsCountList = read_and_divide_input_to_classes(syntheticFile, -1)

# print(len(syntheticFileNew))
# tempNums = [0]*3

split_extra_to_output(neededVals, syntheticDigitsCountList, trainDevTestSizes, syntheticFileNew, outFolder, filesPaths)

# print(totalVals)
# print(neededVals)

allVals = [0]*3
for ind in range(len(totalVals)):
    allVals[ind] = totalVals[ind] + neededVals[ind]
# print(allVals)

extraFile = folder + 'synthetic-extra.csv'
# read extra file and split according to classes
extraFileLen, extraFileNew, extraDigitsCountList = read_and_divide_input_to_classes(extraFile, -1)

extraNeededVals = complete_to_mult(allVals)
dummyProportions = [1]*3

split_extra_to_output(extraNeededVals, extraDigitsCountList, dummyProportions, extraFileNew, outFolder, filesPaths)


# splitted_files_lens=

    # if len(outfileLists[j]) % 100 != 0:
    #     complete_to_mult(currdir, outfilePath, syntheticExtraFile, len(outfileLists[j]) % 100)


