import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.linear_model import LinearRegression
# from mpl_toolkits import mplot3d

from experiments import normalized_ell_1

def successful_rankings_3d_plotter(results, method_name):
    hits = [[0] * 21 for _ in range(21)]
    for experiment in results:
        total = len(experiment) / 100
        for sample in experiment:
            if sample['mu_ranking'] != None:
                hits[int(sample['d_div'] * 20)][int(sample['d_max'] * 20)] += 1 / total
    x = [[i / 20] * 21 for i in range(21)]
    y = [[i / 20 for i in range(21)]] * 21
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, hits, marker='o')
    [ax.plot(x[i], y[i], hits[i], color=(i / 20, 0.0, 0.0)) for i in range(21)]
    ax.set_title('Successfull rankings agains d_div, d_max (' + method_name + ')')
    ax.set_xlabel('d_div')
    ax.set_ylabel('d_max')
    ax.set_zlabel('% of successful rankings')
    plt.savefig(method_name.replace(' ', '_') + '_3d_hits.png')
    plt.close()

def successful_rankings_plotter(results, method_name):
    d_div_hits = [0] * 21
    d_max_hits = [0] * 21
    for experiment in results:
        total = len(experiment) * 21 / 100
        for sample in experiment:
            if sample['mu_ranking'] != None:
                d_div_hits[int(sample['d_div'] * 20)] += 1 / total
                d_max_hits[int(sample['d_max'] * 20)] += 1 / total
    x = [i / 20 for i in range(0, 21)]
    npx = np.array(x).reshape((-1,1))
    npdiv = np.array(d_div_hits)
    div_model = LinearRegression().fit(npx, npdiv)
    div_r_sq = div_model.score(npx, npdiv)
    div_intercept = div_model.intercept_
    div_slope = div_model.coef_
    plt.xlim([-.05, 1.05])
    plt.ylim([0, 105])
    plt.grid()
    plt.plot(x, d_div_hits, 'o')
    plt.plot([0, 1], [div_intercept, div_intercept + div_slope], 'r-', label='Intercept: ' + str(div_intercept) + '\nSlope: ' + str(div_slope[0]) + '\n$R^2$: ' + str(div_r_sq))
    plt.title('Successful rankings against d_div (' + method_name + ')')
    plt.legend()
    plt.xlabel('d_div')
    plt.ylabel('% of successful rankings')
    plt.savefig(method_name.replace(' ', '_') + '_d_div_hits.png')
    plt.close()
    plt.xlim([-.05, 1.05])
    plt.ylim([0, 105])
    plt.grid()
    plt.plot(x, d_max_hits, 'o')
    plt.title('Successful rankings against d_max (' + method_name + ')')
    plt.xlabel('d_max')
    plt.ylabel('% of successful rankings')
    plt.savefig(method_name.replace(' ', '_') + '_d_max_hits.png')
    plt.close()

def ranking_distance_plotter(results, method_name):
    x = [i / 20 for i in range(21)]
    avg_mu_rank_dist = [0.0] * 21
    # avg_div_rank_dist = [0.0] * 21
    max_distance = [0.0] * 21
    min_distance = [1.0] * 21
    # max_div_distance = [0.0] * 21
    # min_div_distance = [1.0] * 21
    total_mus = [0] * 21
    for experiment in results:
        # div_total = len(experiment) * 21
        for sample in experiment:
            crisp_ranking = sample['crisp_ranking']
            # div_distance = normalized_ell_1(crisp_ranking, sample['div_ranking'])
            # avg_div_rank_dist[int(sample['d_max'] * 20)] += div_distance / div_total
            # if div_distance > max_div_distance[int(sample['d_max'] * 20)]:
            #     max_div_distance[int(sample['d_max'] * 20)] = div_distance
            # elif div_distance < min_div_distance[int(sample['d_max'] * 20)]:
            #     min_div_distance[int(sample['d_max'] * 20)] = div_distance
            mu_ranking = sample['mu_ranking']
            if mu_ranking != None:
                distance = normalized_ell_1(crisp_ranking, mu_ranking)
                avg_mu_rank_dist[int(sample['d_max'] * 20)] += distance
                if distance > max_distance[int(sample['d_max'] * 20)]:
                    max_distance[int(sample['d_max'] * 20)] = distance
                elif distance < min_distance[int(sample['d_max'] * 20)]:
                    min_distance[int(sample['d_max'] * 20)] = distance
                total_mus[int(sample['d_max'] * 20)] += 1
    avg_mu_rank_dist = [avg_mu_rank_dist[i] / total_mus[i] for i in range(21)]
    plt.grid()
    plt.plot(x, avg_mu_rank_dist, 'rx-')
    # plt.plot(x, max_distance, 'b-')
    # plt.plot(x, min_distance, 'b-')
    plt.fill_between(x, max_distance, min_distance, color=(0.9, 0.9, 0.9))
    # plt.plot(x, avg_div_rank_dist, 'rx')
    # plt.plot(x, max_div_distance, 'gx')
    # plt.plot(x, min_div_distance, 'yx')
    plt.plot([0, 1], [0, 1], 'g-')
    plt.show()

if __name__ == '__main__':
    with open('shannon_index_results_small.json', 'r') as file:
        results = json.load(file)
    # methods = ['Richness', 'Shannon Index', 'Berger Parker Index', 'Hill numbers', 'Simpson Index']
    methods = ['Shannon Index']
    for method in methods:
        tag = method.lower().replace(' ', '_')
        data_path = tag + '_results_small.json'
        with open(data_path, 'r') as file:
            results = json.load(file)
        # successful_rankings_plotter(results, method)
        # successful_rankings_3d_plotter(results, method)
        ranking_distance_plotter(results, method)