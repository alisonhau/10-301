import csv
import sys
import math
import copy

LEARN_RATE = 0.5
LABELS = [0,1]


def load_dict(dictfile):
    dict_d = {}
    with open(dictfile, 'r') as dictin:
        for word in dictin:
            split_word = word.split(" ")
            dict_d[split_word[1].strip()] = split_word[0]

    print(dict_d)
    return dict_d

def get_dat(infile, lookup):
    unique_words = set()
    dat = []
    with open(infile, 'r') as readin:
        readin = csv.reader(readin, delimiter = '\t')
        for row in readin:
            for word in row[1:]:
                keys = word.split(":")
                decoded = lookup[keys[0]]
                #dat.append(decoded)
                #unique_words.add(decoded)
                dat.append(int(keys[0]))
                unique_words.add(int(keys[0]))

    return dat, unique_words

def dot_prod(theta, x, k, i):
    print(x)
    return theta[k][0] + theta[k][x[i]]

def calc_grad(theta, i, k, xs, ys, K):
    indicator = 1 if ys[i] == k else 0
    num = math.exp(dot_prod(theta, xs, k, i))
    denom = 0
    for j in range(K):
        denom += math.exp(dot_prod(theta, xs, j, i))
    grad = -1 * (indicator - num * 1 / denom)
    return grad

def main():
    train_in = sys.argv[1]
    validation_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epochs = (int) (sys.argv[8])

    dict_d = load_dict(dict_in)
    
    train_dat, train_words = get_dat(train_in, dict_d)
    valid_dat, val_words = get_dat(validation_in, dict_d)
    test_dat, test_words = get_dat(test_in, dict_d)

    all_words = train_words.union(val_words).union(test_words)
    print(all_words)

    grad_train = [0,0]

    train_theta = [ [0 for i in range(len(train_words)+2)] for j in range(len(LABELS)+1)]

    for epoch in range(num_epochs):
        for i in range(len(train_dat)):
            for k in range( len(LABELS) ):
                grad_train[k] = calc_grad(train_theta, i,k, train_dat, LABELS, len(LABELS))

if __name__ == '__main__':
    main()
