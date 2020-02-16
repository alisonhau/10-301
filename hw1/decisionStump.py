import sys

# arguments: 
# <train input .tsv>
# <test input .tsv>
# split index
# train out
# test out
# metrics out



if __name__ == '__main__':
  train_in = sys.argv[1]
  test_in = sys.argv[2]
  split_idx = sys.argv[3]
  train_out = sys.argv[4]
  test_out = sys.argv[5]
  metrics_out = sys.argv[6]
  print("The input file is %s" % (infile))
  print("The output file is %s" % (output))
