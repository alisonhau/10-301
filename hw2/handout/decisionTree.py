import sys
import csv

# arguments:
# train input
# test input
# max depth
# train out
# test out
# metrics out
# 

labels = ['A', 'republican']

train_in = sys.argv[1]
test_in = sys.argv[2]
max_depth = (int)(sys.argv[3])
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]

att = []

cat = []
if 'education' in train_in:
  cat = ['notA', 'A']
  att = ['notA', 'A']
else:
  cat = ['democrat', 'republican']
  att = ['n', 'y']

class Node:

# data -> set
# vote -> if it's a leaf, what's the vote
# left -> left tree (y0)
# right -> right tree (y1)
# split -> idx of splitting attr
# split_label -> str of splitting attr

  def __init__(self, data):
    self.data = data      # set
    self.vote = None
    self.left = None
    self.right = None
    self.split = None
    self.split_label = None
    self.left_size = None
    self.right_size = None

  def get_left_size(self):
    return self.left_size
  def get_right_size(self):
    return self.right_size
  def get_vote(self):
    return self.vote
  def get_left(self):
    return self.left
  def get_right(self):
    return self.right
  def get_data(self):
    return self.data
  def get_split(self):
    return self.split
  def get_split_label(self):
    return self.split_label
  def set_left_size(self, left_size):
    self.left_size = left_size
  def set_right_size(self, right_size):
    self.right_size = right_size
  def set_vote(self, vote):
    self.vote = vote
  def set_left(self, left):
    self.left = left
  def set_right(self, right):
    self.right = right
  def set_data(self,data):
    self.data = data
  def set_split(self, split):
    self.split = split
  def set_split_label(self, split_label):
    self.split_label = split_label

#metrics = {'train': None, 'test': None }
metrics = {'test': 0}

def gini_l(L):
  y0 = 0
  y1 = 0
  tot = 0

  for l in L:
    if l[len(l)-1] in labels:
      y1 += 1
    else:
      y0 += 1
    tot += 1
  return ((y0/tot) * (y1/tot) ) + ((y1/tot) * (y0/tot))

# returns overall gini impurity of dataset
def find_gini(dataset):
  y0 = 0
  y1 = 0
  tot = 0
  for row in dataset:
    last_idx = len(row)-1
    if row[last_idx] in labels:
      y1 += 1
    else:
      y0 += 1
    tot += 1

  return ((y0/tot) * (y1/tot)) + ((y1/tot)*(y0/tot))

def find_split(D,allowed_l, features):
  if len(D) == 0:
    return None
  big_gini = find_gini(D)

  max_gini = 0 
  max_gini_idx = None 

  for idx in allowed_l:
    y0 = []
    y1 = []
    tot = 0
    for row in D:
      tot += 1
      if row[idx] in ['n', 'notA']:
        y0.append(row)
      else:
        y1.append(row)
    gini0 = 0.0 if len(y0) == 0 else gini_l(y0)
    gini1 = 0.0 if len(y1) == 0 else gini_l(y1)
    
    g = big_gini - ( ((len(y0)/tot) * gini0) + ((len(y1)/tot) * gini1) )

    if g >= max_gini:
      max_gini = g
      max_gini_idx = idx

  return max_gini_idx

def majority_vote(D):
  l = len(D[0]) - 1

  votes = {}
  for row in D:
    vote = row[l]
    if vote not in votes:
      votes[vote] = 0
    votes[vote] += 1

  majority = None
  max_freq = -100

  for vote in votes:
    if votes[vote] >= max_freq:
      max_freq = votes[vote]
      majority = vote

  return majority


def train_tree(D, node, depth, features, allowed_idxs):
  if (len(D) == 0):
    node.set_split_label("LEAF -- empty D")
    return node
  
  if len(allowed_idxs) == 0:
    node.set_vote(majority_vote(D))
    node.set_left_size( len([ row[len(D[0])-1] for row in D if row[len(D[0])-1] not in labels] ) )
    node.set_right_size(len(D) - node.get_left_size())

    return node

  if depth <= 0:
    node.set_vote(majority_vote(D))
    node.set_left_size( len([ row[len(D[0])-1] for row in D if row[len(D[0])-1] not in labels] ) )
    node.set_right_size(len(D) - node.get_left_size())
    return node

  if len(set([D[i][len(D[0])-1] for i in range(len(D)) ]) ) == 1: # if perfectly classified
    node.set_vote(majority_vote(D))
    node.set_left_size( len([ row[len(D[0])-1] for row in D if row[len(D[0])-1] not in labels] ) )
    node.set_right_size(len(D) - node.get_left_size())

    return node

  x_m = find_split(D, allowed_idxs, features)
  if x_m == None:
    return node
  
  node.set_split(x_m)
  node.set_split_label(features[x_m])

  D0 = []
  D1 = []
  label0 = []
  label1 = []
  for row in D:
    if row[len(row)-1] in labels:
      label1.append(row)
    else:
      label0.append(row)
    if row[x_m] == 'y' or row[x_m] == 'A':
      D1.append(row)
    else:
      D0.append(row)
  
  new_allowed = [i for i in allowed_idxs if i != x_m]

  new_left = Node(D0)
  node.set_left_size( len(label0) )
  new_left = train_tree(D0, new_left, depth - 1, features, new_allowed)
  node.set_left(new_left)

  new_right = Node(D1)
  node.set_right_size( len(label1) )
  new_right = train_tree(D1, new_right, depth - 1, features, new_allowed)
  node.set_right(new_right)

  return node

def train(D, stop_depth):
  d_l = []
  with open(D) as tsvin:
    tsvin = csv.reader(tsvin, delimiter = "\t")
    features = next(tsvin)
    for row in tsvin:
      d_l.append(row)
  root = Node(d_l)
  allowed = [i for i in range(len(features)-1)]
  return train_tree(d_l, root, stop_depth, features, allowed)

def h(r, root):
  if root.get_split() == None:
    return root.get_vote()
  
  if r[root.get_split()] in ['y', 'A']:
    return h(r, root.get_right())
  else:
    return h(r, root.get_left())

def get_h(xs, dt, out):
  with open(xs) as xsin:
    with open(out, 'w') as xsout:
      xsin = csv.reader(xsin, delimiter = '\t')
      next(xsin)
      err = 0
      tot = 0
      for row in xsin:
        res = h(row, dt)
        if res != row[len(row)-1]:
          err += 1
        tot += 1
        xsout.write("%s\n"% res)

      if 'test' in xs:
        metrics['test'] = err/tot
      elif 'train' in xs:
        metrics['train'] = err/tot

def print_metrics(outfile):
  print("%s\t" % (metrics['test']))
  return
  with open(outfile, 'w') as fileout:
    fileout.write("error(%s): %f\n" % ('train', metrics['train']))
    fileout.write("error(%s): %f\n" % ('test', metrics['test']))

def pretty_print(tree, depth, max_depth):
  if tree == None:
    return
  print("[%s %s / %s %s]" % (tree.get_left_size(), cat[0],tree.get_right_size(), cat[1]), end = '')
  if tree.get_split_label() == None:
    return
  if depth == max_depth: return
  

  print("\n%s %s = n: " % ("|" * (depth + 1), tree.get_split_label()), end = '')
  pretty_print(tree.get_left(), depth+1, max_depth)
  print("\n%s %s = y: " % ("|" * (depth + 1), tree.get_split_label()), end = '')
  pretty_print(tree.get_right(), depth+1, max_depth)
  return 

def pretty_print_wrap(tree, depth, max_depth):
  pretty_print(tree, depth, max_depth)
  print()

if __name__ == '__main__':
  
  root = train(train_in, max_depth);
  get_h(train_in, root, train_out);
  get_h(test_in, root, test_out);
  pretty_print_wrap(root, 0, max_depth)
  print_metrics(metrics_out)

