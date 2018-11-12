import csv
import os
import numpy as np

trainDevTestSizes = [0.7, 0.15, 0.15]  # [0.6, 0.2, 0.2]  #


def complete_to_mult(currdir, filepath, extra_file_path, need_to_complete):
    os.chdir(currdir)

    with open(extra_file_path, newline='') as csvfile:
        dbreader = csv.reader(csvfile)

    return


parFolder = 'C:/Users/H/Documents/Haifa Univ/Thesis/deep-learning/deep-input/exp-and-synthetic-with-grayscale'
# hidden/gray/'
# exp-and-synthetic-with-grayscale/'#-total-2429/'

folder = parFolder + 'data/'
outFolder = 'data-divided-by-100'  # 'data-hidden-gray'#'data-with-grayscale/' #'./data'

currdir = os.getcwd()
# print (currdir)
os.chdir(folder)
# os.chdir(folder)


dataSize = 2429 #2500 #2200
totalVals = [0]*3
lens = [0]*3
neededVals = [0]*3
series = 'exp-synthetic'
filesPaths = []
for it in ['train', 'dev', 'test']:
    path = '/{}/'.format(it) + '{}-data-{}.csv'.format(series, it)
    filesPaths.append(path)


# filesPaths = ['/train/' + 'exp-synthetic-data-train.csv', '/dev/' + 'exp-synthetic-data-dev.csv', '/test/' + 'exp-synthetic-data-test.csv']
# filesPaths = ['/train/' + 'hidden-gray-data-train.csv', '/dev/' + 'hidden-gray-data-dev.csv', '/test/' + 'hidden-gray-data-test.csv']
# filesPaths = ['/train/' + 'exp-data-train.csv', '/dev/' + 'exp-data-dev.csv', '/test/' + 'exp-data-test.csv']

for file in os.listdir(folder):
    # print(file)
    filename = os.path.splitext(file)[0]
    if filename == 'synthetic' or filename == 'synthetic-extra':
        continue
    if file.endswith('txt'):
        continue

    infile = folder + file
    outfileNew = []
    infileLen = 0
    digitsCountList = [[] for k2 in range(10)]
    outFiles = [[] for k3 in range(3)]  # will contain mat of values for train / dev / test

    with open(infile, newline='') as csvfile:
        dbreader = csv.reader(csvfile)
        for row in dbreader:
            color = row[0:24]
            digit = row[-1]
            # digit = row[-2]
            tempDigit = int(digit)
            if tempDigit == -1:  # ignore samples with "no digit" tag
                continue

            newRow = []
            newRow.extend(color)
            newRow.append(digit)

            # save rows indexes of label = i
            digitsCountList[tempDigit].append(infileLen)
            infileLen += 1  # save current file length

            outfileNew.append(newRow)

    # print(digitsCountList)

    for i in range(10):  # iterate digits 0-9
        currList = digitsCountList[i]
        listLen = len(currList)

        for j in range(3):  # iterate over train/ dev/ test
            if j < 2:
                lens[j] = int(round(trainDevTestSizes[j] * listLen))
            else:
                lens[2] = listLen - lens[0] - lens[1]
            totalVals[j] += lens[j]

            if lens[j] > 0:
                # choose samples with label = i from available samples and delete chosen samples
                chosenList = np.random.choice(currList, lens[j], replace=False)
                currList = [item for item in currList if item not in chosenList]
                # save chosen sample in matrix that will be written to compatible output file
                for el in chosenList:
                    outFiles[j].append(outfileNew[el])

    for j in range(3):  # iterate over train/ dev/ test

        # print('filename: {} j: {} len: {}, totalVal {}'.format(filename, j, len(outFiles[j]),totalVals[j]))

        os.chdir(currdir)

        outfilePath = outFolder + str(filesPaths[j])
        # print(outfilePath)
        # write collected data to output file

        with open(outfilePath, 'a+', newline='') as csvfile:
            mywriter = csv.writer(csvfile, delimiter=',')
            for outrow in outFiles[j]:
                mywriter.writerow(outrow)


# uncomment in case splitting data from few files(experiment data)
# and need to add more data from another file named "synthetic"

for j in range(3):
    # calculate how many missing samples should be taken from the 'extra file'
    neededVals[j] = int(trainDevTestSizes[j]*dataSize) - totalVals[j]

print(neededVals)

outfileNew = []
extraFileLen = 0
digitsCountList = [[] for k1 in range(10)]
outfileLists = [[] for m in range(3)]

syntheticFile = folder + 'synthetic.csv'
with open(syntheticFile, newline='') as csvfile:
    dbreader = csv.reader(csvfile)
    for row in dbreader:
        digit = row[-1]
        tempDigit = int(digit)

        # divide extra file's indexes according to labels (digits)
        digitsCountList[tempDigit].append(extraFileLen)
        extraFileLen += 1

        outfileNew.append(row)

print(len(outfileNew))
tempNums = [0]*3
# tempNums = [[0]*3]*10
for j in range(3):  # for train/ dev / test
    for i in range(10):  # for every digit possible
        currList = digitsCountList[i]
        listLen = len(currList)
        if i < 9:
            # for j in range(3):  # for train/ dev / test

            # calculate how many samples are needed from a certain digit
            extraListLen = int(round(trainDevTestSizes[j] * listLen))
            if tempNums[j] + extraListLen < neededVals[j]:
                tempNums[j] += extraListLen
            else:
                extraListLen = neededVals[j] - tempNums[j]
                tempNums[j] = neededVals[j]

        else:  # i==9
            extraListLen = neededVals[j] - tempNums[j]

        if extraListLen > 0:
            if extraListLen < len(currList):
                chosenList = np.random.choice(currList, extraListLen, replace=False)
            else:
                chosenList = currList
            currList = [item for item in currList if item not in chosenList]
            for el in chosenList:
                outfileLists[j].append(outfileNew[el])


# outfileNew = []
# extraFileLen = 0
# digitsCountList = [[] for k1 in range(10)]
# syntheticExtraFile = folder + 'synthetic-extra.csv'
# with open(syntheticExtraFile, newline='') as csvfile:
#     dbreader = csv.reader(csvfile)
#     for row in dbreader:
#         digit = row[-1]
#         tempDigit = int(digit)
#
#         # divide extra file's indexes according to labels (digits)
#         digitsCountList[tempDigit].append(extraFileLen)
#         extraFileLen += 1
#
#         outfileNew.append(row)

for j in range(3):  # iterate over train/ dev/ test
    os.chdir(currdir)
    outfilePath = outFolder + str(filesPaths[j])

    print('j: {} len: {}'.format(j, len(outfileLists[j])))

    with open(outfilePath, 'a+', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        for outrow in outfileLists[j]:
            mywriter.writerow(outrow)

    # if len(outfileLists[j]) % 100 != 0:
    #     complete_to_mult(currdir, outfilePath, syntheticExtraFile, len(outfileLists[j]) % 100)


