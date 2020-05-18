import numpy as np
a = np.array([ [0,0,0,0,0,0,0],
            [0,1,1,1,1,1,0],
            [0,1,0,0,1,0,0],
            [0,1,0,1,0,0,0],
            [0,1,1,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0] ])

a = np.array([ [1,1,0,0,0],
                [1,1,1,0,0],
                [0,1,1,1,0],
                [0,0,1,1,1],
                [0,0,0,1,1] ])

weights = np.array([ [-1,0,1],
                    [-1,0,1],
                    [-1,0,1] ])

def ip(a, b):
    ret = 0
    for row in range(len(a)):
        for col in range(len(a[0])):
            ret += a[row][col] * b[row][col] 

    return ret


soln = np.array([ [0 for _ in range(len(a[0])-2)] for _ in range(len(a)-2)] )

for i in range(len(a) - len(weights) + 1):
    for j in range(len(a[0]) - len(weights[0])+ 1):
        b = a[i:i+len(weights), j:j+len(weights)]
        #soln[i][j] = sum(sum(np.inner(weights, b)))
        soln[i][j] = ip(weights,b)

print(a)
print(weights)
print(soln)

