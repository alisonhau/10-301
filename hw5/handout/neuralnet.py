# alison hau (ahau) #
# March 2020 #
# 1-hidden-layer NN #

# imports #
import numpy as np
from sys import argv
from random import random
import csv
from math import exp,log

label_maps = { 0 : 'a',
                1 : 'e',
                2 : 'g',
                3 : 'i',
                4 : 'l',
                5 : 'n',
                6 : 'o',
                7 : 'r',
                8 : 't',
                9 : 'u'
                }

def activation1(a):
    return 1/(1 + np.exp(-a))

def activation2(b):
    return np.divide(np.exp(b), np.sum(np.exp(b)))

def get_in(input_file):
    dat = np.loadtxt(input_file, delimiter=',')
    y = dat[:, 0]
    x = np.copy(dat)
    x[:,0] = 1

    return x,y.astype(int)

def init_weights(flag,r,c):
    bias = np.zeros((r,1))
    if flag == 1:   # random
        a = np.random.uniform(-0.1, 0.10000001, (r, c))
    elif flag == 2: # zero
        a = np.zeros((r,c))
    #else:
        #print("Init_flag %i not valid" % flag)
    return np.append(bias,a,1)

def LinearForward(x,alpha):
    return np.dot(np.transpose(alpha),x)

def SigmoidForward(a):
    z = np.array(activation1(a) )
    return np.append(z, [1])

def SoftmaxForward(b):
    return activation2(b)

def XEntropyForward(y,yhat):
    zeros = np.zeros(len(yhat))
    zeros[y] = -1 * np.log(yhat[y])
    return zeros

def NNForward(x,y,alpha,beta):
    #print("FORWARD")
    #print("alpha: %s" % alpha)
    #print("beta: %s" % beta)
    a = LinearForward(x,alpha)
    #print("a: %s" % a)
    z = SigmoidForward(a)
    #print("z: %s" % z)
    b = LinearForward(z,beta)
    #print("b: %s" % b)
    yhat = SoftmaxForward(b)
    #print("yhat: %s" % yhat)
    J = XEntropyForward(y,yhat)
    #print("J: %s" % J[y])

    return (x,a,z,b,yhat,J)

def XEntropyBackward(y,yhat,J,gJ):
    zeros = np.zeros(len(yhat))
    y_val = -1.0 / yhat[y]
    zeros[y] = y_val
    return zeros

def SoftmaxBackward(y,yhat,gyhat):
    sub = np.subtract(np.diag(yhat), np.outer(yhat, np.transpose(yhat)))
    trans = np.transpose(gyhat)
    return np.dot(trans, sub)

def LinearBackward(z,b,gb,beta):
    gbeta = np.transpose(np.outer( np.transpose(gb), np.transpose(z)))
    gz = np.dot( beta, gb)
    return gbeta, gz

def SigmoidBackward(a,z,gz):
    fromone = np.vectorize(lambda x: 1-x)
    sub = fromone(z)
    mult = np.multiply(z, sub)
    mult2 = np.multiply(gz, mult)
    return mult2[1:]

def NNBackward(x,y,alpha,beta,o):
    #print("BACKWARD")
    (x,a,z,b,yhat,J) = o
    gJ = 1
    gyhat = XEntropyBackward(y,yhat,J,gJ)
    #print("gyhat (entback): %s" % gyhat)
    gb = SoftmaxBackward(y,yhat,gyhat)
    #print("gb (softmax): %s" % gb)
    gbeta, gz = LinearBackward(z,b,gb,beta)
    #print("gbeta(linback): %s" % gbeta)
    #print("gz (linback): %s" % gz)
    ga = SigmoidBackward(a,z,gz)
    #print("ga (sigback): %s" % ga)
    galpha,x = LinearBackward(x,a,ga,alpha)
    #print("galpha (linback): %s" % galpha)
    
    return galpha, gbeta

def output_metrics(s):
    with open(metrics_out, 'w') as writeout:
        writeout.write(s)

def SGD(xtrain, ytrain, xtest, ytest):
    metrics_string = ""
     # init weights
    num_features = len(xtrain[0])-1
    num_classes = len(label_maps)
    alpha = np.transpose(init_weights(init_flag, hidden_units, num_features))
    beta = np.transpose(init_weights(init_flag, num_classes, hidden_units)  )
   

    for i in range(num_epoch):
        #print("\n\nEPOCH %s\n" % i)
        J_tot = []
        test_J_tot = []

        yhats_l = []
        test_yhats_l = []
        for rownum in range(len(xtrain)):
            #print("SAMPLE %s" % rownum)
            x = xtrain[rownum]
            y = ytrain[rownum]
            #print("x: %s" % x)
            #print("y: %s" % y)
            # compute NN layers
            o = NNForward(x,y,alpha,beta)
            (x,a,z,b,yhat,J) = o
            #print("YHATS: ",yhat)
            # compute gradients via backprop
            g_alpha, g_beta = NNBackward(x,y,alpha,beta,o)
            # update params
            alpha -= learning_rate * g_alpha
            beta -= learning_rate * g_beta
        
        for rownum in range(len(xtest)):
            x = xtest[rownum]
            y = ytest[rownum]
            # compute NN layers
            test_o = NNForward(x,y,alpha,beta)
            (_,_,_,_,yhat,test_J) = test_o
            test_yhats_l.append(np.argmax(yhat))
            test_J_tot.append(test_J[y])

        # eval training mean cross-entropy
        for rownum in range(len(xtrain)):
            x = xtrain[rownum]
            y = ytrain[rownum]
            # compute NN layers
            train_o = NNForward(x,y,alpha,beta)
            (_,_,_,_,yhat,J) = train_o
            yhats_l.append(np.argmax(yhat))
            J_tot.append(J[y])

        #print(J_tot)
        #print(test_J_tot)
        xent = (sum(J_tot)/len(J_tot))
        test_xent = (sum(test_J_tot)/len(test_J_tot))
        metrics_string += "epoch=%s crossentropy(%s): %f\n" % (i+1, 'train', xent)
        metrics_string += "epoch=%s crossentropy(%s): %f\n" % (i+1, 'test', test_xent)

    testright = 0
    trainright = 0
    
    with open(train_out, "w") as trainout:
        for i in range(len(ytrain)):
            trainout.write("%s\n" % yhats_l[i])
            #print(ytrain[i], yhats_l[i])
            if ytrain[i] == yhats_l[i]:
                trainright += 1
    with open(test_out, "w") as testout:
        for i in range(len(ytest)):
            testout.write("%s\n" % test_yhats_l[i])
            if ytest[i] == test_yhats_l[i]:
                testright += 1

    #print(test_yhats_l)
    #print(ytest)
    #print(testright, len(ytest))
    trainerr =  1 - trainright/len(ytrain)
    testerr =  1 - testright/len(ytest)

    trainerr_s = "error(%s): %.2f\n" % ("train", trainerr)
    testerr_s = "error(%s): %.2f\n" % ("test", testerr)
    metrics_string += trainerr_s + testerr_s
     
    output_metrics(metrics_string)

    return alpha, beta

def main():
    # get train x and y and dim
    X_train, Y_train = get_in(train_input)

    X_test, Y_test = get_in(test_input)

    # SGD
    alpha, beta = SGD(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    train_input = argv[1]
    test_input = argv[2]
    
    train_out = argv[3]
    test_out = argv[4]
    metrics_out = argv[5]

    num_epoch = int(argv[6])
    hidden_units = int(argv[7])

    init_flag = int(argv[8])

    learning_rate = float(argv[9])

    main()

