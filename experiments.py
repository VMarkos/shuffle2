import math
import json
import random
import diversity_metrics

from timeit import default_timer as timer

# def shuffle2(sample, d_max, d_div, diversity_metric, crisp_ranking):
#     X = get_points(sample)
#     class_distribution = sample['classCount']
#     mu_rank = non_recursive_shuffle2(X, class_distribution, crisp_ranking, diversity_metric, d_div, d_max, diversity_metric) #TODO Check if you need the entire sample!
#     return mu_rank

def dist(mu, D_mu):
    return abs(mu - D_mu)

def sort_by(mu, D_mu, X, mu_rank, classes):
    distances = {x: dist(mu(get_distribution(mu_rank + [x], classes)), D_mu) for x in X if x not in mu_rank}
    return list({x: val for x, val in sorted(distances.items(), key=lambda item: item[1], reverse=True)}.keys())

def get_distribution(points, classes):
    distribution = [0] * len(classes)
    for point in points:
        distribution[classes[str(point[0]) + ',' + str(point[1])]] += 1
    return distribution

"""
Non recursive Shuffle2:
front: a stack, initially contains a ranking of X according to SortBy.
muRank = empty list
While (front != empty && len(X) > len(muRank)):
    next_el = front.pop()
    muRank.append(next_el)
    if ell_1(rank, muRank) > d_max:
        muRank.pop()
    else:
        front += sortBy(mu, D_mu, X \ muRank)
"""

def shuffle2(sample, d_max, d_div, diversity_metric, crisp_ranking): # NON-recursive version of Shuffle2 algorithm
    X = get_points(sample)
    classes = sample['classes']
    mu_rank = []
    front = sort_by(diversity_metric, d_div, X, mu_rank, classes)
    while (len(front) > 0 and len(X) > len(mu_rank)):
        next_el = front.pop()
        mu_rank.append(next_el)
        if normalized_ell_1(crisp_ranking, mu_rank) > d_max:
            mu_rank.pop()
        else:
            front += sort_by(diversity_metric, d_div, X, mu_rank, classes)
    if len(X) == len(mu_rank):
        return mu_rank
    return None

def update_class_distribution(X, mu_rank, classes):
    distribution = [0] * len(classes)
    for x in X:
        if x not in mu_rank:
            distribution[classes[str(x[0]) + ',' + str(x[1])]] += 1
    return distribution

# def unpack_str_to_tuple(string):
#     items = string.split(',')
#     return (float(items[0]), float(items[1]))

# def recursive_shuffle2(X, class_distribution, ranking, mu, D_mu, d_max, front, mu_rank):
#     if len(X) == len(mu_rank):
#         return
#     if len(front) == 0:
#         return False
#     next_el = front.pop()
#     mu_rank.append(next_el)
#     if normalized_ell_1(ranking, mu_rank) > d_max:
#         popped = mu_rank.pop()
#         recursive_shuffle2(X, class_distribution, ranking, mu, D_mu, d_max, front, mu_rank)
#     else:
#         front += sort_by(mu, D_mu, [x for x in X if x not in mu_rank], class_distribution)
#         recursive_shuffle2(X, class_distribution, ranking, mu, D_mu, d_max, front, mu_rank)

def normalized_ell_1(x, y):
    n = len(x)
    return 1 / math.ceil(0.5 * (n ** 2 - 1)) * ell_1(x, y)

def ell_1(x, y):
    n = min(len(x), len(y))
    if n < 1:
        return 0
    perm = find_permutation(x, y)
    return sum([abs(i - perm[i]) for i in range(n)])

def find_permutation(rank_1, rank_2): # IMPORTANT! Assumes that rank_1 has the most points in it!
    n = min(len(rank_1), len(rank_2))
    return [rank_1.index(rank_2[i]) for i in range(n)]

def ell_2(x, y):
    return sum([abs(x[i] - y[i]) ** 2 for i in range(len(x))])

def ell_2_rank(target, xs, ys):
    distances = {}
    for i in range(len(xs)):
        point = (xs[i], ys[i])
        distances[point] = ell_2(target, point)
    return list({x: val for x, val in sorted(distances.items(), key=lambda item: item[1])}.keys())

def experiment(diversity_metric, data, d_div, d_max):
    results = []
    # i = 0
    for sample in data:
        # print(i)
        # i += 1
        target = sample['target']
        crisp_ranking = ell_2_rank(target, sample['x'], sample['y'])
        div_ranking = sort_by(diversity_metric, d_div, get_points(sample), [], sample['classes'])
        start = timer()
        mu_ranking = shuffle2(sample, d_max, d_div, diversity_metric, crisp_ranking)
        end = timer()
        exec_time = end - start
        results.append({
            'exec_time': exec_time,
            'crisp_ranking': crisp_ranking,
            'mu_ranking': mu_ranking,
            'div_ranking': div_ranking,
            'd_div': d_div,
            'd_max': d_max,
        })
        # print(target)
        # print(crisp_ranking)
    return results

def get_points(sample):
    points = []
    xs = sample['x']
    ys = sample['y']
    for i in range(len(xs)):
        points.append((xs[i], ys[i]))
    return points

def set_of_experiments(diversity_metric, metric_id):
    print(metric_id)
    data = []
    with open('small_data/dataset.json', 'r') as file:
        data = json.load(file)
    results = []
    for d_div in range(0,21):
        print('d_div: ', d_div)
        for d_max in range(0,21):
            # print('d_max: ', d_max)
            results.append(experiment(diversity_metric, data, d_div / 20, d_max / 20))
    with open(metric_id + '_results_small.json', 'w') as file:
        json.dump(results, file, indent=2)

if __name__ == '__main__':
    # data = []
    # with open('small_data/dataset.json', 'r') as file:
    #     data = json.load(file)
    # d_div = 0.2
    # d_max = 0.1
    # results = experiment(diversity_metrics.shannon_index, data[:10], d_div, d_max)
    # with open('RESULTS.json', 'w') as file:
    #     json.dump(results, file, indent=2)
    set_of_experiments(diversity_metrics.richness, 'richness')
    set_of_experiments(diversity_metrics.shannon_index, 'shannon_index')
    set_of_experiments(diversity_metrics.simpson_index, 'simpson_index')
    set_of_experiments(diversity_metrics.berger_parker_index, 'berger_parker_index')
    set_of_experiments(diversity_metrics.hill_numbers, 'hill_numbers')