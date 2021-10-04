import random
import json

import math
from matplotlib import pyplot as plt

def generate_samples(M, n_total, path):
    centers = [(random.random(), random.random()) for _ in range(M)]
    xs = []
    ys = []
    class_distribution = []
    classes = {}
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    partitions = generate_partition(n_total, M)
    for i in range(len(centers)):
        center = centers[i]
        n = partitions[i]
        class_distribution.append(n)
        max_r = min([max(abs(center[0] - x[0]), abs(center[1] - x[1])) / 2 for x in centers if x != center] + [center[0], center[1], 1 - center[0], 1 - center[1]])
        (x, y) = generate_blob(center, max_r, n)
        for j in range(len(x)):
            classes[str(x[j]) + ',' + str(y[j])] = i
        plt.plot(x, y, 'o')
        xs += x
        ys += y
    target = (random.random(), random.random())
    plt.plot(target[0], target[1], 'x')
    plt.suptitle('Size: ' + str(n_total) + ', partition: ' + str(partitions), fontsize=24, y=.95)
    plt.title('x: target point\ndifferent colors denote different classes', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path + '.png')
    plt.close()
    return {
        'x': xs,
        'y': ys,
        'M': M,
        'size': n_total,
        'target': target,
        'classCount': class_distribution,
        'classes': classes,
    }

def generate_partition(n, k): # Split n into the sum of k non-negative integers (uniform random)
    splitters = sorted(random.sample([x for x in range(0, n)], k - 1))
    partition = [splitters[0]]
    for i in range(1, k - 1):
        partition.append(splitters[i] - splitters[i - 1])
    partition.append(n - splitters[-1])
    return partition

def generate_blob(center, max_r, n):
    x = []
    y = []
    for i in range(n):
        dx = 2 * max_r * random.random() - max_r
        dy = 2 * max_r * random.random() - max_r
        x.append(center[0] + dx)
        y.append(center[1] + dy)
    return (x, y)

def generate_dataset(N, M, n_total):
    samples = []
    print('Size: ', n_total)
    for i in range(N):
        # print('Iteration: ' + str(i))
        path = 'fixed_count_data/sample_' + str(i) + '_size_' + str(n_total)
        samples.append(generate_samples(M, n_total, path))
    with open('fixed_count_data/dataset_n_total=' + str(n_total) + '.json', 'w') as file:
        json.dump(samples, file, indent=2)

if __name__ == '__main__':
    N = 10
    M = 5
    for n_total in range(15, 100, 5):
        generate_dataset(N, M, n_total)
