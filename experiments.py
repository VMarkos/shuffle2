import math
import json
import random
import diversity_metrics

from timeit import default_timer as timer

def shuffle2(sample, d_max, d_div, diversity_metric, crisp_ranking):
    X = get_points(sample)
    class_distribution = sample['classCount']
    div_rank = []
    front = sort_by(diversity_metric, d_div, X, class_distribution)
    recursive_shuffle2(X, class_distribution, crisp_ranking, diversity_metric, d_div, d_max, front, div_rank) #TODO Check if you need the entire sample!
    return div_rank

def dist(mu, D_mu):
    return abs(mu - D_mu)

def sort_by(mu, D_mu, X, distribution):
    distances = {}
    for item in X:
        distances[item] = dist(mu(distribution), D_mu)
    return list({x: val for x, val in sorted(distances.items(), key=lambda item: item[1], reverse=True)}.keys())

def recursive_shuffle2(X, class_distribution, ranking, mu, D_mu, d_max, front, mu_rank):
    if len(X) == len(mu_rank):
        return
    if len(front) == 0:
        return False
    next_el = front.pop()
    mu_rank.append(next_el)
    if normalized_ell_1(ranking, mu_rank) > d_max:
        popped = mu_rank.pop()
        recursive_shuffle2(X, class_distribution, ranking, mu, D_mu, d_max, front, mu_rank)
        return
    front.append(sort_by(mu, D_mu, [x for x in X if x not in mu_rank], class_distribution))
    recursive_shuffle2(X, class_distribution, ranking, mu, D_mu, d_max, front, mu_rank)

def normalized_ell_1(x, y):
    n = len(x)
    return 1 / math.ceil(0.5 * (n ** 2 - 1)) * ell_1(x, y)

def ell_1(x, y):
    n = min(len(x), len(y))
    if n < 1:
        return 0
    print(x, y)
    perm = find_permutation(x, y)
    return sum([abs(i - perm[i]) for i in range(n)])

def find_permutation(rank_1, rank_2): # IMPORTANT! Assumes that rank_1 has the most points in it!
    n = min(len(rank_1), len(rank_2))
    print(len(rank_1), len(rank_2))
    return [rank_1.index(rank_2[i]) for i in range(n)]

def ell_2(x, y):
    return sum([abs(x[i] - y[i]) ** 2 for i in range(len(x))])

def ell_2_rank(target, xs, ys):
    distances = {}
    for i in range(len(xs)):
        point = (xs[i], ys[i])
        # print(target, point)
        distances[point] = ell_2(target, point)
    return list({x: val for x, val in sorted(distances.items(), key=lambda item: item[1], reverse=True)}.keys())

def experiment(diversity_metric, data, d_div, d_max):
    results = []
    # print(data)
    for sample in data:
        # print(sample)
        target = sample['target']
        crisp_ranking = ell_2_rank(target, sample['x'], sample['y'])
        start = timer()
        mu_ranking = shuffle2(sample, d_max, d_div, diversity_metric, crisp_ranking)
        end = timer()
        exec_time = end - start
        results.push({
            'exec_time': exec_time,
            'crisp_ranking': crisp_ranking,
            'mu_ranking': mu_ranking,
            'd_div': d_div,
            'd_max': d_max,
        })
    return results

def get_points(sample):
    points = []
    xs = sample['x']
    ys = sample['y']
    for i in range(len(xs)):
        points.append((xs[i], ys[i]))
    return points

if __name__ == '__main__':
    data = []
    with open('data/dataset.json', 'r') as file:
        data = json.load(file)
    d_div = 0.6
    d_max = 0.4
    results = experiment(diversity_metrics.shannon_index, data, d_div, d_max)
    with open('temp_results.json', 'w') as file:
        json.dump(results, file, indent=2)