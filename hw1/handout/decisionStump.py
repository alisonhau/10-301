import sys
import csv

# arguments: 
# <train input .tsv>
# <test input .tsv>
# split index
# train out
# test out
# metrics out

mv = [0] * 2
metrics = {'test': 0, 'train': 0}

def majority_vote(D):
  l = len(D[0]) - 1

  votes = {}
  for data in D:
    vote = data[l]
    if vote not in votes:
      votes[vote] = 0
    votes[vote] += 1

  majority = None
  max_freq = -100

  for vote in votes:
    if votes[vote] >= max_freq:
      max_freq = votes[vote]
      majority = vote
  print(votes)
  return majority
  

def train(D, s):
  with open(train_in) as tsvin:
    tsvin = csv.reader(tsvin, delimiter = '\t')
    D0 = []
    D1 = []
    next(tsvin)
    for row in tsvin:
      if row[s] == 'y' or row[s] == 'A':
        D1.append(row)
      else:
        D0.append(row)
    
    mv[0] = majority_vote(D0)
    mv[1] = majority_vote(D1)
    
    print(mv)

def get_h(xs, s, out):
  results = []
  total = 0
  errors = 0
  with open(xs) as xsin:
    xsin = csv.reader(xsin, delimiter = '\t')
    next(xsin)
    for row in xsin:
      total += 1
      if row[s] == 'y' or row[s] == 'A':
         results.append(mv[1])
         if row[len(row)-1] != mv[1]:
           errors += 1
      else:
        results.append(mv[0])
        if row[len(row)-1] != mv[0]:
          errors += 1

  with open(out, 'w') as fileout:
    for result in results:
      fileout.write(result)
      fileout.write('\n')

  # metrics
  if 'test' in xs:
    metrics['test'] = errors/total
  elif 'train' in xs:
    metrics['train'] = errors/total
  

def print_metrics(outfile):
  print(metrics)
  with open(outfile, 'w') as fileout:
    fileout.write("error(%s): %f\n" % ('train', metrics['train']))
    fileout.write("error(%s): %f\n" % ('test', metrics['test']))

if __name__ == '__main__':
  train_in = sys.argv[1]
  test_in = sys.argv[2]
  split_idx = (int)(sys.argv[3])
  train_out = sys.argv[4]
  test_out = sys.argv[5]
  metrics_out = sys.argv[6]

  train(train_in, split_idx);
  get_h(train_in, split_idx, train_out);
  get_h(test_in, split_idx, test_out);

  print_metrics(metrics_out)

