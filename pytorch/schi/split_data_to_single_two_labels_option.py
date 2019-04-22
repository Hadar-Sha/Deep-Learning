import csv
import os
import numpy as np
import argparse

# NUM_OF_CLASSES = 10
# NUM_OF_OUTPUT_FILES = 3
# CLASS = 9

OPTIONS_LIST = [[2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [9, 0],
                [2, 1], [5, 1], [6, 1],
                [0, 2], [1, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [9, 2],
                [0, 3], [2, 3], [4, 3], [5, 3], [6, 3],
                [0, 4], [2, 4], [3, 4], [5, 4], [6, 4], [7, 4],
                [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [7, 5],
                [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [7, 6], [9, 6],
                [2, 7], [4, 7], [5, 7], [6, 7],
                [0, 9], [2, 9], [6, 9]]

# EASY_OPTIONS = [9, 5]
EASY_OPTIONS = [[8, 0],
                [0, 1], [3, 1], [8, 1], [9, 1],
                [8, 2],
                [8, 3], [9, 3],
                [8, 4], [9, 4],
                [8, 5], [9, 5],
                [8, 6],
                [0, 7], [3, 7], [8, 7], [9, 7],
                [8, 9]]

parser = argparse.ArgumentParser()
parser.add_argument('--inFolder', default='./data/data-two-labels-big'
                    , help="parent Directory containing input files")

parser.add_argument('--subfolder', default='data/', help="Directory containing input files")

parser.add_argument('--baseOutFolder', default='./data/data-two-labels-big'
                    , help="parent Directory containing input files")
parser.add_argument('--isEasy', default=False, type=bool)

if __name__ == '__main__':

    currdir = os.getcwd()
    args = parser.parse_args()
    options = EASY_OPTIONS if args.isEasy else OPTIONS_LIST

    for i in range(len(options)):
        currOption = options[i]
        outFolder = args.baseOutFolder +'-only-{}-{}'.format(options[i][0], options[i][1])
        if not os.path.isdir(outFolder):
            os.chdir(os.path.dirname(outFolder))
            os.mkdir(os.path.basename(outFolder))

        os.chdir(currdir)

        for it in ['train', 'dev', 'test']:
            path = os.path.join(args.inFolder, it)

            out_data = []
            file_names = []

            for file in os.listdir(path):  # should be only one .csv file
                file_names.append(file)
                with open(os.path.join(path, file), 'r', newline='') as csv_file:
                    my_reader = csv.reader(csv_file, delimiter=',')
                    for row in my_reader:
                        digits_vec = list(map(int, row[-2:]))
                        if digits_vec == currOption:
                            out_data.append(row)

            out_path = os.path.join(outFolder, it)
            if not os.path.isdir(out_path):
                os.chdir(os.path.dirname(out_path))
                os.mkdir(os.path.basename(out_path))
            else:
                os.chdir(outFolder)

            os.chdir(it)
            for out_file in file_names:
                with open(out_file, 'a+', newline='') as out_csv_file:
                    my_writer = csv.writer(out_csv_file, delimiter=',')
                    for out_row in out_data:
                        my_writer.writerow(out_row)

            os.chdir(currdir)

