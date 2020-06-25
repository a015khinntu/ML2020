import subprocess
import numpy as np

# seed = int(numpy.random.randint(0, 1000, 1))

for seed in range(1000):
    subprocess.call('python gradient_discent.py --num_feat=10 --iter=300 --lr=0.5 --seed={}'.format(seed))
