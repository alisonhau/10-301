import re
import csv
import sys
import math
import copy

MODEL2_THRESHOLD = 4

def parse_dict(infile):
    d = dict()
    idx = 0
    with open(infile, 'r') as readin:
        for row in readin:
            line = row.split(" ")
            d[line[0]] = int(line[1])
            idx += 1
    
    return d

def model1(dict_d, datafile, outfile):
    ret_dat = []
    with open(datafile, 'r') as infile:
        with open(outfile,'w') as out:
            infile = csv.reader(infile, delimiter = '\t')
            for row in infile:
                seen = set()
                label = row[0]
                text = re.split(" ",row[1])
                out.write(label)
                for word in text:
                    if word in dict_d and word not in seen:
                        seen.add(word)
                        out.write('\t' + str(dict_d[word]) + ':' + '1')
                out.write('\n')

def get_counts(data):
    counts_d = {}
    with open(data, 'r') as datain:
        datain = csv.reader(datain, delimiter = '\t')
        for row in datain:
            text = re.split(" ", row[1])
            for word in text:
                if word not in counts_d:
                    counts_d[word] = 0
                counts_d[word] += 1

    return counts_d

def model2(dict_d, datafile, outfile):
    counts_d = get_counts(datafile)
    print(counts_d)
    ret_dat = []
    with open(datafile, 'r') as infile:
        with open(outfile,'w') as out:
            infile = csv.reader(infile, delimiter = '\t')
            for row in infile:
                seen = set()
                label = row[0]
                text = re.split(" |,|\.|--|\_ ",row[1])
                outstr = label
                for word in text:
                    if word in dict_d and text.count(word) < MODEL2_THRESHOLD and word not in seen:
                        seen.add(word)
                        outstr += "".join(['\t',str(dict_d[word]),':','1'])
                outstr += '\n'
                out.write(outstr)


def main():
    train_in = sys.argv[1]
    validation_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_val_out  = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = (int) (sys.argv[8])
    
    dict_d = parse_dict(dict_in)
    
    if feature_flag == 1:
        train_dat = model1(dict_d,train_in, formatted_train_out)
        valid_dat = model1(dict_d, validation_in, formatted_val_out)
        test_dat = model1(dict_d, test_in, formatted_test_out)
    else:
        train_dat = model2(dict_d,train_in, formatted_train_out)
        valid_dat = model2(dict_d, validation_in, formatted_val_out)
        test_dat = model2(dict_d, test_in, formatted_test_out)    

    #print(train_dat, empties)
    #print(valid_dat)
    #print(test_dat)


if __name__ == '__main__':
    main()

def get_dat(infile):
    dat = []
    empty_lines = []
    with open(infile, 'r') as readin:
        readin = csv.reader(readin, delimiter = '\t')
        for row in readin:
            dat.append(row)
            if not row:
                empty_lines.append(len(dat)-1)

    return dat, empty_lines
