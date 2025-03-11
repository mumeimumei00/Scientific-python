import numpy as np
import time
import os
import matplotlib.pyplot as plt 

# First Method to create Pseudo Random Number: Manually and Sequentially

def randomTime():
    t = time.perf_counter()
    input("Please enter to get a random number:")
    return 1000000000*t- int(100000000*t)*10


randomdistri = np.array([])
for x in range(200):
    r = randomTime()
    randomdistri = np.append(randomdistri, r)

plt.hist(randomdistri)
plt.show()



