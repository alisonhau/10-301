import csv
import sys
import math
import copy

def main():
  #initial values
  feature_flag = 1


  train_input = sys.argv[1] #.tsv
  validation_input = sys.argv[2] #.tsv
  test_input = sys.argv[3] #.tsv
  dict_in = sys.argv[4]
  train_output = sys.argv[5]
  test_output = sys.argv[6] #.labels
  metrics_output = sys.argv[6] #.txt
  num_epoch = int(sys.argv[7]) #number of times each training ex used in SGD
  learn_rate = 0.5

  train_info = []
  blank_spaces = []
  with open(train_input, 'rb') as input_file:
    reader = csv.reader(input_file, delimiter='\t')
    for row in reader:
      train_info.append(row)
      if not row:
        blank_spaces.append(len(train_info)-1)

  valid_info = []
  with open(validation_input, 'rb') as input_file:
    reader = csv.reader(input_file, delimiter='\t')
    for row in reader:
      valid_info.append(row)
 
  test_info = []
  test_blank_spaces = []
  with open(test_input, 'rb') as input_file:
    reader = csv.reader(input_file, delimiter='\t')
    for row in reader:
      test_info.append(row)
      if not row:
        test_blank_spaces.append(len(test_info)-1)

  #theta = k x (m+1), *_data = n x (m+1), theta (k) = 1 x (m+1), *_data (i) = 1 x (m+1)
  #where m = # unique words, k = # unique labels, n = total # training samples
  #*_data (i)  = [1 . . . ] for bias, but its weight in theta (k) = 0
  if feature_flag == 1:
    word_dict, label_dict, train_data, train_label, theta = parse_model_one(train_info)
  else:
    word_dict, label_dict, train_data, train_label, theta = parse_model_two(train_info)
  
  
  valid_data = []
  valid_label = []
  for elems in valid_info:
    if elems:
      valid_data.append(word_dict[elems[0]])
      valid_label.append(label_dict[elems[1]])
    else: 
      if feature_flag == 2:
        valid_data.append(-1)
        valid_label.append(-1)

  test_data = []
  test_label = []
  for elems in test_info:
    if elems:
      test_data.append(word_dict[elems[0]])
      test_label.append(label_dict[elems[1]])
    else:
      if feature_flag == 2:
        test_data.append(-1)
        test_label.append(-1)
  
  metrics_output = open(metrics_output, 'w')

  gradients = [0 for a in range(0, len(label_dict))]
  gradients_valid = [0 for a in range(0, len(label_dict))]
  gradients_test = [0 for a in range(0, len(label_dict))]
  for e in range(0, num_epoch):
    for i in range(0, len(train_data)):
      for k in range(0, len(label_dict)):
        gradients[k] = calc_grad(theta, i, k, train_data, train_label, len(label_dict), feature_flag)
      for k in range(0, len(label_dict)):
        if feature_flag == 1:
          theta = sgd_m1_step(theta, i, k, learn_rate, train_data, gradients)
        else:
          theta = sgd_m2_step(theta, i, k, learn_rate, train_data, gradients)
    
    metrics_output.write("epoch=" + str(e+1) + " likelihood(train): " + str(calc_likelihood(theta, train_data, train_label, len(label_dict), feature_flag)) + "\n")
    metrics_output.write("epoch=" + str(e+1) + " likelihood(validation): " + str(calc_likelihood(theta, valid_data, valid_label, len(label_dict), feature_flag)) + "\n")


  #predictions for training data
  space_index = 0
  train_output = open(train_output, 'w')
  train_output_vals = []
  for i in range(0, len(train_data)):
    if feature_flag == 1:
      if space_index < len(blank_spaces) and i == blank_spaces[space_index] - space_index:
        train_output.write("\n")
        space_index+=1
        train_output_vals.append("")
      prediction = calc_predictions(theta, train_data, label_dict, i, len(label_dict), feature_flag)
      train_output.write(prediction + "\n")
      train_output_vals.append(prediction)

    else:
      if train_data[i] == -1:
        train_output.write("\n")
        train_output_vals.append("")
      else:  
        prediction = calc_predictions(theta, train_data, label_dict, i, len(label_dict), feature_flag)
        train_output.write(prediction + "\n")
        train_output_vals.append(prediction)
  train_output.close()
  
  num_train_errors = 0
  for i in range(0, len(train_info)):
    if train_info[i]:
      if train_info[i][1] != train_output_vals[i]:
        num_train_errors += 1
  metrics_output.write("error(train): " + str(num_train_errors * 1.0 / (len(train_info) - len(blank_spaces))) + "\n")


  #predictions for test data
  test_space_index = 0
  test_output = open(test_output, 'w')
  test_output_vals = []
  for i in range(0, len(test_data)):
    if feature_flag == 1:
      if test_space_index < len(test_blank_spaces) and i == test_blank_spaces[test_space_index] - test_space_index:
        test_output.write("\n")
        test_space_index+=1
        test_output_vals.append("")
      prediction = calc_predictions(theta, test_data, label_dict, i, len(label_dict), feature_flag)
      test_output.write(prediction + "\n")
      test_output_vals.append(prediction)
    else:
      if test_data[i] == -1:
        test_output.write("\n")
        test_output_vals.append("")
      else:
        prediction = calc_predictions(theta, test_data, label_dict, i, len(label_dict), feature_flag)
        test_output.write(prediction + "\n")
        test_output_vals.append(prediction)
  test_output.close()
  
  num_test_errors = 0
  for i in range(0, len(test_info)):
    if test_info[i]:
      if test_info[i][1] != test_output_vals[i]:
        num_test_errors += 1
  metrics_output.write("error(test): " + str(num_test_errors * 1.0 / (len(test_info) - len(test_blank_spaces))) + "\n")
  metrics_output.close()


def parse_model_one(train_info):
  x = []
  y = []
  label_dict = {}
  word_dict = {}
  for elems in train_info:
    #checks non-empty
    if elems:
      if elems[1] not in label_dict:
        y.append(len(label_dict))
        label_dict[elems[1]] = len(label_dict)
      else:
        y.append(label_dict[elems[1]])
      if elems[0] not in word_dict:
        x.append(len(word_dict) + 1)
        word_dict[elems[0]] = len(word_dict) + 1
      else:
        x.append(word_dict[elems[0]])
  k = len(label_dict) # unique labels
  m = len(word_dict)  # unique words
  theta = [[0 for i in range(m+2)] for j in range(k+1)] 
  return word_dict, label_dict, x, y, theta

def parse_model_two(train_info):
  x = []
  y = []
  label_dict = {}
  word_dict = {}
  for elems in train_info:
    #checks non-empty
    if elems:
      if elems[1] not in label_dict:
        y.append(len(label_dict))
        label_dict[elems[1]] = len(label_dict)
      else:
        y.append(label_dict[elems[1]])
      if elems[0] not in word_dict:
        x.append(len(word_dict) + 1)
        word_dict[elems[0]] = len(word_dict) + 1
      else:
        x.append(word_dict[elems[0]])
    else:
      x.append(-1)
      y.append(-1)
  k = len(label_dict) # unique labels
  m = len(word_dict)  # unique words
  theta = [[0 for i in range(3*m+4)] for j in range(k+1)] 
  return word_dict, label_dict, x, y, theta




#computed for all k before updating any individual theta (k)
def sgd_m1_step(theta, i, k, learn_rate, x, gradients):
  theta[k][0] = theta[k][0] - learn_rate * gradients[k]
  theta[k][x[i]] = theta[k][x[i]] - learn_rate * gradients[k]
  return theta

def sgd_m2_step(theta, i, k, learn_rate, x, gradients):
  theta[k][0] = theta[k][0] - learn_rate * gradients[k]
  m = (len(theta[0]) - 4) / 3 
  theta[k][x[i]+1+m] = theta[k][x[i]+1+m] - learn_rate * gradients[k]
  if i == 0 or x[i-1] == -1:
    theta[k][1] = theta[k][1] - learn_rate * gradients[k] #BOS
  else:
    theta[k][x[i-1]+1] = theta[k][x[i-1]+1] - learn_rate * gradients[k]
  if i == (len(x)-1) or x[i+1] == -1:
    theta[k][len(theta[0])-1] = theta[k][len(theta[0])-1] - learn_rate * gradients[k] #EOS
  else:
    theta[k][x[i+1]+1+2*m] = theta[k][x[i+1]+1+2*m] - learn_rate * gradients[k]


  return theta


def calc_grad(theta, i, k, x, y, K, feature_flag):
  indicator = 1 if y[i] == k else 0
  numerator = math.exp(calc_dot_prod(theta, x, k, i, feature_flag)) 
  denominator = 0
  for j in range(0, K):
    denominator += math.exp(calc_dot_prod(theta, x, j, i, feature_flag)) 
  gradient = -(indicator - numerator * 1.0 / denominator)
  return gradient

def calc_predictions(theta, x, label_dict, i, K, feature_flag):
  probs = []
  for j in range(0, K):
    probs.append(math.exp(calc_dot_prod(theta, x, j, i, feature_flag))) 
  max_index = probs.index(max(probs))
  ord_dict = dict((val, key) for key, val in label_dict.iteritems())
  return ord_dict[max_index]
  

def calc_likelihood(theta, x, y, K, feature_flag):
  sum = 0
  for i in range(0, len(x)):
    for k in range(0, K):
      indicator = 1 if y[i] == k else 0
      inner_sum = 0
      exp = math.exp(calc_dot_prod(theta, x, k, i, feature_flag)) 
      for k2 in range(0, K):
        inner_sum += math.exp(calc_dot_prod(theta, x, k2, i, feature_flag)) 
      sum += indicator * math.log(exp * 1.0 / inner_sum)
  return - sum * 1.0 / len(filter(lambda a: a != -1, x)) 

def calc_dot_prod(theta, x, k, i, feature_flag):
  if feature_flag == 1:
    return theta[k][0] + theta[k][x[i]]
  else:
    if x[i] == -1: return 0
    sum = 0
    m = (len(theta[0]) - 4) / 3
    sum += theta[k][0] + theta[k][x[i] + 1 + m]
    if i == 0 or x[i-1] == -1:
      sum += theta[k][1]  #BOS
    else:
      sum += theta[k][x[i-1]+1]  
    if i == (len(x)-1) or x[i+1] == -1:
      sum += theta[k][len(theta[0])-1]  #EOS
    else:
      sum += theta[k][x[i+1]+1+2*m]
    return sum

main()
