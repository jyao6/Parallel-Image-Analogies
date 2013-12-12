import numpy as np
import math
import matplotlib.pyplot as plt

# num_pixels = map(math.log, [253*384, 362*300, 373*350,430*650,768*1024,1200*1600])
# num_pixels =  map(lambda x:x, [253*384, 362*300, 373*350,430*650,768*1024,1200*1600])
# runtimes = [21.2011690044,23.7003680661,23.6705871205,53.0915109361,141.33235601,346.651560904]
# red dashes, blue squares and green triangles
# plt.plot(num_pixels, runtimes, 'ro-')
# plt.title('Parallel Runtime vs Image Size (4 Cores, Blur Analogy)')
# plt.ylabel('Time (seconds)')
# plt.xlabel('Total Number of Pixels')
# plt.show()

num_cores = range(1,5,1)
runtimes = [186.872838004,94.9083470415,67.8254259596,53.0915109361]
plt.plot(num_cores, runtimes, 'ro-')
plt.title('Parallel Runtime vs Number of Cores (Blur Analogy, Miley)')
plt.ylabel('Time (seconds)')
plt.xlabel('Number of Cores')
plt.show()