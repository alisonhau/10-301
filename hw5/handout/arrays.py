import numpy as np
import math

alpha = np.array([ [1,1,2,-3,0,1,-3],
                        [1,3,1,2,1,0,2],
                        [1,2,2,2,2,2,1],
                        [1,1,0,2,1,-2,2] ])

alpha_star = np.array([ [1,2,-3,0,1,-3],
                        [3,1,2,1,0,2],
                        [2,2,2,2,2,1],
                        [1,0,2,1,-2,2] ])

beta = np.array([ [1,1,2,-2,1],
                        [1,1,-1,1,2],
                        [1,3,1,-1,1] ])

beta_star = np.array([ [1,2,-2,1],
                        [1,-1,1,2],
                        [3,1,-1,1] ])

x = np.transpose(np.array([1,1,1,0,0,1,1]))
y = np.transpose(np.array([0,1,0]))

z0 = x

def activation1(aj):
    return 1/(1 + math.exp(-aj))

def activation2_num(bk):
    return math.exp(bk)

def activation2_tot(arr):
    return sum(math.exp(bl) for bl in arr)

def loss_fn(yhat, ystar):
    tot = 0
    for i in range(3):
        tot += ( y[i] * math.log(yhat[i]))
    return (-1)*tot

print(alpha[1])

u1 = np.dot(alpha,z0)
print('u(1) = %s' % (u1))

z1 = np.array( [activation1(ui) for ui in u1] )
print('z(1) = %s' % (z1) )


u2 = np.dot(beta_star,z1)
print('u(2) = %s' % (u2))

act2_denom = activation2_tot(u2)
z2 = np.array( [activation2_num(ui)/act2_denom for ui in u2] )
print('z(2) = %s' % (z2) )

loss = loss_fn(z2, y)
print('loss = %s' % loss)

yhat = z2
# back

gl = 1
gy = np.transpose(np.array([ (-1*y[0])/(yhat[0]) , (-1*y[1]/yhat[1]) , (-1*y[2]/yhat[2]) ]))
print("gy: %s " % gy)

gb = np.transpose(np.array( [ yhat[0]-y[0], yhat[1]-y[1], yhat[2]-y[2] ] ))
print("gb: %s " % gb)

gz = np.matmul(gb,beta_star)
print("gz: %s" % gz)

gbeta = np.array( [[ gb[i] * z1[j] for j in range(len(z1))] for i in range(len(gb))])
print('gbeta: %s' % gbeta)

ga = np.array([gz[i] * (1 + math.exp(-1*u1[i]))*(math.exp(-1*u1[i])) for i in range(len(u1))])
print('ga: %s' % ga)

print(ga)
print(x)
galpha = np.array( [[ga[i] * x[j] for j in range(len(x))] for i in range(len(ga))])
print('galpha: %s' % galpha)

STEPSIZE = 1

print("1: %s" % (gbeta[0,0]))
print("2: %s" % (beta[0,0] - STEPSIZE * gbeta[0,0]))
print("3: %s" % (alpha[2,4] - STEPSIZE * galpha[2,4]))
print("4: %s" % (alpha[1,0] - STEPSIZE * galpha[1,0]))
print("5: %s" % (3))

