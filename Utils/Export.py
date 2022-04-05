import csv
from datetime import date
from os.path import exists

def makeFile(filename):
    f = open(filename, 'x')

def checkIfFileExists(filename):
    return exists(filename)

def getCsvHeader(filename):
    colNames = []
    with open(filename, 'r') as csv_file:
        for row in csv_file:
            colNames = row
            break
    return colNames.replace("\n", "").split(",")

def appendCsv(filename, string):
    with open(filename, 'a') as csv_file:
        csv_file.write("\n" + ",".join(string))

def setCsvHeader(filename, header):
    with open(filename, 'w') as csv_file:
        csv_file.write(",".join(header))


def saveData(filename, data):
    if not checkIfFileExists(filename):
        makeFile(filename)
        setCsvHeader(filename, [k for k in data])
    
    csvHeader = getCsvHeader(filename)

    missingCol = []
    for column in csvHeader:
        if column not in data:
            missingCol.append(column)
    if len(missingCol) > 0:
        print("Error: Column mismatch (" + ",".join([str(k) for k in missingCol]) + " in csv but not in data to append)")
        return False
    
    missingCol = []
    for column in data:
        if column not in csvHeader:
            missingCol.append(column)
    if len(missingCol) > 0:
        print("Error: Column mismatch (" + ",".join([str(k) for k in missingCol]) + " in data to append but not in csv)")
        return False

    appendCsv(filename, [str(data[k]) for k in csvHeader])