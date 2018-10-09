import csv
import os
import numpy as np

parFolder = 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/with-user-study-labels/'
folder = parFolder + 'without-counters/'
# 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/with-user-study-labels/without-counters/'
outFolder = parFolder + 'for-neural-net/'
os.chdir(folder)


for file in os.listdir(folder):
    filename = os.path.splitext(file)[0]
    infile = folder + file
    outfileNew = []
    infileLen = 0
    digitsCountList = [[] for k in range(10)]
    outfileTrain = []
    outfileTest = []
    outfileValidation = []



    with open(infile, newline='') as csvfile:
        dbreader = csv.reader(csvfile)
        for row in dbreader:
            ### change row so that only colors and labels will remain in data!!!
            color = row[0:24]
            digit = row[-2]
            tempDigit = int(digit)
            if tempDigit == -1: # ignore samples with "no digit" tag
                continue
                # tempDigit= 10
                # digit= str(tempDigit)
            newRow = []
            newRow.extend(color)
            newRow.append(digit)

            digitsCountList[tempDigit].append(infileLen)
            infileLen += 1

            outfileNew.append(newRow)
            # outfileNew.append(row)

    for i in range(10):
        currList = digitsCountList[i]
        listLen = len(currList)
        trainLen = round(0.8 * listLen)
        testLen = round(0.15 * listLen)
        valLen = listLen - trainLen - testLen

        if trainLen > 0:
            chosenTrain = np.random.choice(currList,trainLen,replace=False)
            currList = [item for item in currList if item not in chosenTrain]
            for el in chosenTrain:
                outfileTrain.append(outfileNew[el])

        if testLen > 0:
            chosenTest = np.random.choice(currList, testLen, replace=False)
            currList = [item for item in currList if item not in chosenTest]
            for el in chosenTest:
                outfileTest.append(outfileNew[el])

        if valLen > 0:
            chosenVal = np.random.choice(currList, valLen, replace=False)
            currList = [item for item in currList if item not in chosenVal]
            for el in chosenVal:
                outfileValidation.append(outfileNew[el])

    # np.random.permutation(outfileNew)
    # trainSize = round(0.8*infileLen)
    # testSize = round(0.15*infileLen)
    # valSize = infileLen-trainSize-testSize

    # outfileTrain= outfileNew[0:trainSize]
    # outfileTest= outfileNew[trainSize:trainSize+testSize]
    # outfileValidation= outfileNew[trainSize+testSize:infileLen]

    outfileTrainPath = outFolder + 'exp-data-train.csv'
    outfileTestPath = outFolder + 'exp-data-test.csv'
    outfileValidationPath = outFolder + 'exp-data-validation.csv'

    with open(outfileTrainPath, 'a+', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        for outrow in outfileTrain:
            mywriter.writerow(outrow)

    with open(outfileTestPath, 'a+', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        for outrow in outfileTest:
            mywriter.writerow(outrow)

    with open(outfileValidationPath, 'a+', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        for outrow in outfileValidation:
            mywriter.writerow(outrow)
