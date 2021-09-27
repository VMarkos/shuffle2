import random
import json

import math
from matplotlib import pyplot as plt

def generate_samples(M, n_min, n_max, path):
    centers = [(random.random(), random.random()) for i in range(M)]
    xs = []
    ys = []
    n_total = 0
    class_distribution = []
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    for center in centers:
        n = random.randint(n_min, math.floor(n_min + (n_max - n_min) / 2)) + random.randint(0, math.ceil((n_max - n_min) / 2))
        n_total += n
        class_distribution.append(n)
        max_r = min([max(abs(center[0] - x[0]), abs(center[1] - x[1])) / 2 for x in centers if x != center] + [center[0], center[1], 1 - center[0], 1 - center[1]])
        (x, y) = generate_blob(center, max_r, n)
        plt.plot(x, y, 'o')
        xs += x
        ys += y
    target = (random.random(), random.random())
    plt.plot(target[0], target[1], 'x')
    plt.suptitle('Size: ' + str(n_total) + ', n_min: ' + str(n_min) + ', n_max: ' + str(n_max), fontsize=24, y=.95)
    plt.title('x: target point\ndifferent colors denote different classes', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path + '.png')
    plt.close()
    return {
        'x': xs,
        'y': ys,
        'M': M,
        'nMin': n_min,
        'nMax': n_max,
        'size': n_total,
        'target': target,
        'classCount': class_distribution,
    }

def generate_blob(center, max_r, n):
    x = []
    y = []
    for i in range(n):
        dx = 2 * max_r * random.random() - max_r
        dy = 2 * max_r * random.random() - max_r
        x.append(center[0] + dx)
        y.append(center[1] + dy)
    return (x, y)

def generate_dataset(N, M, n_min, n_max):
    samples = []
    for i in range(N):
        print('Iteration: ' + str(i))
        path = 'data/sample_' + str(i)
        samples.append(generate_samples(M, n_min, n_max, path))
    with open('data/dataset.json', 'w') as file:
        json.dump(samples, file, indent=2)

if __name__ == '__main__':
    N = 5 #FIXME This is 100! --- not factorial...
    M = 10
    n_min = 0
    n_max = 30
    generate_dataset(N, M, n_min, n_max)