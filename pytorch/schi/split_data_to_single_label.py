import csv
import os
import numpy as np
import argparse

NUM_OF_CLASSES = 10
NUM_OF_OUTPUT_FILES = 3
CLASS = 9

parser = argparse.ArgumentParser()
parser.add_argument('--inFolder', default='./data/data-with-grayscale-4000'
                    , help="parent Directory containing input files")

parser.add_argument('--subfolder', default='data/', help="Directory containing input files")
parser.add_argument('--outFolder', default='./data/data-w-gray-only-9', help="Directory containing output files")


if __name__ == '__main__':

    currdir = os.getcwd()
    args = parser.parse_args()
    if not os.path.isdir(args.outFolder):
        os.chdir(os.path.dirname(args.outFolder))
        os.mkdir(os.path.basename(args.outFolder))

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
                    digit = int(row[-1])
                    if digit == CLASS:
                        out_data.append(row)

        out_path = os.path.join(args.outFolder, it)
        if not os.path.isdir(out_path):
            os.chdir(os.path.dirname(out_path))
            os.mkdir(os.path.basename(out_path))

        os.chdir(it)
        for out_file in file_names:
            with open(out_file, 'a+', newline='') as out_csv_file:
                my_writer = csv.writer(out_csv_file, delimiter=',')
                for out_row in out_data:
                    my_writer.writerow(out_row)

        os.chdir(currdir)


