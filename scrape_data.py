# Takes a comma delineated string into a list ofto a list of floats and one int
def lineToList(line):
    ls = line.split(",")
    ls = [float(x) for x in ls]
    ls[-1] = int(ls[-1]) # Set the last element to a list
    return ls

def getData(fileName):
    infile = open(fileName)
    lst = infile.readlines()
    return [lineToList(line) for line in lst]
