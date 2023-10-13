import numpy as np
import random
def prob(x):
    return

def gaussian():
    return

x_0 = []
chain = []
stop = False
x_i = x_0
while(stop == False):
    p_i = prob(x_i)
    x_j = x_i + gaussian()
    p_j = prob(x_j)
    ratio = p_j / p_i
    if ratio >=1:
        x_i = x_j
    else:
        u = random.random()
        if ratio > u:
            x_i = x_j
        else:
            x_i = x_i
    chain.append(chain, x_i)

##at some point stop = True

