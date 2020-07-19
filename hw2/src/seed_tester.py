import subprocess
import numpy as np

for i in np.random.randint(0, 1000, 100):
    subprocess.call('python logistic.py --seed={}'.format(i))

