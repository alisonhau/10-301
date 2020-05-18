import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

X = np.array( [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

y = np.array([ [0],
                [1],
                [1],
                [0] ])

np.random.seed(1)

#randomly init weights w/ mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.ranodm((4,1)) - 1

for j in xrange(6000):
    # feedfwd
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss target?
    l2_error = y - l2

    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))

    # what dir is target val?
    l2_delta = l2_error*nonlin(l2,deriv=True)

    #hwo much did each l1 val contribute to l2 error (acc to weights)?
    l1_error = l2_delta.dot(syn1.t)

    # backprop
    # in what dir is target l1
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn2 += l0.T.dot(l1_delta)
