# calculates overall gini impurity (before splits) and error rate (percent inforrectly classified instances)

import sys
import csv

# arguments: 
# input
# output

# grade -- A, notA
# party -- republican, democrat
labels = ['A', 'republican']
  
# return gini impurity of dataset at in_file
def find_gini(in_file):
  with open(in_file) as tsvin:
    tsvin = csv.reader(tsvin, delimiter = '\t')
    next(tsvin)
    y0 = 0
    y1 = 0
    tot = 0
    for row in tsvin:
      last_idx = len(row)-1
      if row[last_idx] in labels:
        y1 += 1
      else:
        y0 += 1   # notA or n
      tot += 1
    
    # calc gini impurity
    ret = ((y0/tot) * (y1/tot)) + ((y1/tot) * (y0/tot))

    return ret

# return error rate of dataset at in_file
def find_err(in_file):
  with open(in_file) as tsvin:
    tsvin = csv.reader(tsvin, delimiter = '\t')
    y0 = 0
    y1 = 0
    next(tsvin)
    for row in tsvin:
      if row[len(row)-1] in labels:
        y1 += 1
      else:
        y0 += 1
    if y1 > y0:
      return y0 / (y1 + y0)
    else:
      return y1 / (y1 + y0)

def print_metrics(outfile, G, E):
  with open(outfile, 'w') as fileout:
    fileout.write("gini_impurity: %f\n" % (G))
    fileout.write("error: %f\n" % (E))

if __name__ == '__main__':
  in_file= sys.argv[1]
  out_file = sys.argv[2]
  
  gini = find_gini(in_file)
  err = find_err(in_file)

  print_metrics(out_file, gini, err)

