import math
import sys
import numpy as np

def parse_tok(tok):
    splitted = tok.split("_")

    return (splitted[0], splitted[1])

def parse_lines(filein):
    words = set()
    tags = set()
    wordmat = []
    tagmat = []
    assocdict = {}
    with open(filein, 'r') as readin:
        readlines = readin.readlines()
    
    for line in readlines:
        wordline = []
        tagline = []
        for word in line.split(" "):
            (w, t) = parse_tok(word.strip())
            words.add(w)
            tags.add(t)
            wordline.append(w)
            tagline.append(t)
            if (w,t) not in assocdict:
                assocdict[(w,t)] = 0
            assocdict[(w,t)] += 1
        wordmat.append(wordline)
        tagmat.append(tagline)

    return (wordmat, tagmat, assocdict)

def to_idxs(rawlist, idxfile):
    converted_list = []

    with open(idxfile, 'r') as idxin:
        idxlist = idxin.readlines()
        idxlist = list(map(lambda x: x.strip(), idxlist))
        idxdict = {k:v for v,k in enumerate(idxlist)}
    
    for sent in rawlist:
        converted_list.append(list(map(lambda x: idxdict[x], sent)))

    return (converted_list, idxdict)

def log_sum_exp(x):
    m = max(x)

    tot = 0
    for xi in x:
        tot += math.exp(xi - m)
    return m + math.log(tot)

if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    prior_mat = np.loadtxt(hmmprior)
    emit_mat = np.loadtxt(hmmemit)
    trans_mat = np.loadtxt(hmmtrans)

    log_it = np.vectorize(lambda x: math.log(x))
    prior_mat = log_it(prior_mat)
    emit_mat = log_it(emit_mat)
    trans_mat = log_it(trans_mat)

    train_words, train_tags, assocdict = parse_lines(train_input)
    train_word_idxs, worddict = to_idxs(train_words, index_to_word)
    train_tag_idxs, tagdict = to_idxs(train_tags, index_to_tag)

    rtagdict = {tagdict[k]:k for k in tagdict}
    rworddict = {worddict[k]:k for k in worddict}

    correct = 0
    total = 0
    log_lik = 0

    outstring = "" 

    for entry in range(len(train_word_idxs)):
        x = train_word_idxs[entry]
        print(x)
        K = len(tagdict)
        T = len(x)

        # fwd
        alpha = [[]]
        for j in range(0,K):
            alpha[0].append((prior_mat[j] + emit_mat[j][x[0]]))
        for t in range(1,T):
            arow = [] 
            for k in range(K):
                emit = emit_mat[k][x[t]]
                sm = []
                for j in range(K):
                    sm.append( alpha[t-1][k] + trans_mat[j][k])
                arow.append(log_sum_exp(sm) + emit)
            alpha.append(arow)

        # bwd
        beta = [[]]
        for i in range(K):
            beta[0].append(0)
        for t in reversed(range(T-1)):
            brow = []
            for k in range(K):
                sm = [] 
                for j in range(K):
                    sm.append( emit_mat[j][x[t+1]] + beta[-1][j] + trans_mat[k][j])
                brow.append(log_sum_exp(sm))
            beta.append(brow)

        print(beta)
        beta = list(reversed(beta))
        print(beta)
        
        alpha = np.array(alpha)
        beta = np.array(beta)

        print(alpha)
        exp_it = np.vectorize(lambda x: math.exp(x))
        p_yts = np.array([np.add(exp_it(alpha[t]), exp_it(beta[t])) for t in range(len(alpha))])
        
        yt_hats = [np.where(t == np.amax(t))[0][0] for t in p_yts]
        print(p_yts) 
        y_actuals = train_tag_idxs[entry]

        if yt_hats == y_actuals:
            correct += 1
        total += 1


        fullwords = train_words[entry]
        pred = []
        for i in range(len(fullwords)):
            pred.append("%s_%s" % (fullwords[i], rtagdict[yt_hats[i]]))

        print(" ".join(pred))
        outstring += " ".join(pred) + "\n"

        loc_log_lik = log_sum_exp(alpha[T-1])
        print(loc_log_lik)
        log_lik += loc_log_lik

    accuracy = correct / total
    print("Train Accuracy: %s" % accuracy)

    print("Train Average Log-Likelihood: %s" % (log_lik / total) )
    
    with open(metric_file, 'w') as metout:
        metout.write("Average Log-Likelihood %s" % (log_lik / total) )

    with open(predicted_file, 'w') as predout:
        predout.write(outstring)
