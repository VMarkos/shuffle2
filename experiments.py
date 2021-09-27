import math

def experiment(data, d_max, d_div, diversity_metric, crisp_ranking):
    div_rank = []
    front = sort_by(diversity_metric, d_div, data)
    recursive_rank(front, div_rank)
    return div_rank

def dist(mu, D_mu):
    return abs(mu - D_mu)

def sort_by(mu, D_mu, X):
    distances = {}
    for item in X:
        distances[item] = dist(mu, D_mu)
    return [x for x in sorted(distances.items(), key=lambda item: item[1]).values()]

def recursive_rank(X, ranking, mu, D_mu, d_max, front, mu_rank):
    if len(X) == len(mu_rank):
        return
    if len(front) == 0:
        return False
    next_el = front.pop()
    mu_rank.append(next_el)
    if normalized_ell_1(ranking, mu_rank) > d_max:
        pass

def normalized_ell_1(x, y):
    n = len(x)
    return 1 / math.ceil(0.5 * (n^2 - 1)) * ell_1(x, y)

def ell_1(x, y):
    n = min(len(x), len(y))
    if n < 1:
        return 0
    return sum([abs(x[i] - y[i]) for i in range(n)])
