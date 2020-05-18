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

if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    train_words, train_tags, assocdict = parse_lines(train_input)
    train_word_idxs, worddict = to_idxs(train_words, index_to_word)
    train_tag_idxs, tagdict = to_idxs(train_tags, index_to_tag)

    # calc priors
    pi_num = np.array([])
    for tag in range(len(tagdict)):
        firsttags = [x[0] for x in train_tag_idxs]
        pi_num = np.append(pi_num, firsttags.count(tag) + 1)

    pidenom = sum(pi_num) 
    pi = np.array( list(map( lambda x: (x) / (pidenom), pi_num) ) )


    # write priors
    with open(hmmprior, 'w') as priorout:
        np.savetxt(priorout, pi)

    # calc As
    a_num = []
    for j in range(len(tagdict)):
        arow = []
        for k in range(len(tagdict)):
            adj = [ [ (row[i] == j and row[i+1] == k) for i in range(len(row) - 1) ] for row in train_tag_idxs]
            arow.append( sum( x.count(True) for x in adj )  + 1)
        
        a_num.append(arow)

    asums = [sum(a_num[i]) for i in range(len(a_num))]

    a = [ [a_num[j][k] / asums[j] for k in range(len(a_num))] for j in range(len(a_num)) ]
    a = np.array(a)

    # write As
    with open(hmmtrans, 'w') as transout:
        np.savetxt(transout, a)

    # calc Bs
    rworddict = {worddict[k]:k for k in worddict}
    rtagdict = {tagdict[k]:k for k in tagdict}
    bnum = []
    for j in range(len(tagdict)):
        brow = []
        for k in range(len(worddict)):
            jtag = rtagdict[j]
            kword = rworddict[k]
            if (kword, jtag) in assocdict:
                counts = assocdict[(kword, jtag)] + 1
            else:
                counts = 1
            brow.append(counts)
        bnum.append(brow)

    
    bsums = [ sum(bnum[i]) for i in range(len(bnum)) ]

    b = np.array( [[ bnum[j][k] / bsums[j] for k in range(len(bnum[0]))] for j in range(len(bnum))] )
    
    # write As
    with open(hmmemit, 'w') as emitout:
        np.savetxt(emitout, b)

