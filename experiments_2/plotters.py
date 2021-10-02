import matplotlib.pyplot as plt
import numpy as np
import json
import math

from sklearn.linear_model import LinearRegression
# from mpl_toolkits import mplot3d

from experiments2 import normalized_ell_1
import diversity_metrics

def successful_rankings_3d_plotter(results, method_name):
    hits = [[0] * 21 for _ in range(21)]
    for experiment in results:
        total = len(experiment) / 100
        for sample in experiment:
            if sample['mu_ranking'] != None and sample['d_max'] != 0:
                hits[int(sample['d_div'] * 20)][int(sample['d_max'] * 20)] += 1 / total
    x = [[i / 20] * 21 for i in range(21)]
    y = [[i / 20 for i in range(21)]] * 21
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, hits, marker='o')
    [ax.plot(x[i], y[i], hits[i], color=(i / 20, 0.0, 0.0)) for i in range(21)]
    ax.set_title('% of complete shufflings against $d_{div}$ and $d_{max}$ (' + method_name + ')')
    ax.set_xlabel('$d_{div}$')
    ax.set_ylabel('$d_{max}$')
    ax.set_zlabel('% of complete shufflings')
    plt.savefig(method_name.replace(' ', '_') + '_3d_hits.png')
    plt.close()

def successful_rankings_plotter(results, method_name):
    d_div_hits = [0] * 21
    d_max_hits = [0] * 21
    for experiment in results:
        total = len(experiment) * 21 / 100
        for sample in experiment:
            if sample['mu_ranking'] != None and sample['d_max'] != 0:
                d_div_hits[int(sample['d_div'] * 20)] += 1 / total
                d_max_hits[int(sample['d_max'] * 20)] += 1 / total
    x = [i / 20 for i in range(0, 21)]
    npx = np.array(x).reshape((-1,1))
    npdiv = np.array(d_div_hits)
    div_model = LinearRegression().fit(npx, npdiv)
    div_r_sq = div_model.score(npx, npdiv)
    div_intercept = div_model.intercept_
    div_slope = div_model.coef_
    # Ranking agains div
    plt.xlim([-.05, 1.05])
    plt.ylim([0, 105])
    plt.grid()
    plt.plot(x, d_div_hits, 'o')
    plt.plot([0, 1], [div_intercept, div_intercept + div_slope], 'r-', label='Intercept: ' + str(div_intercept) + '\nSlope: ' + str(div_slope[0]) + '\n$R^2$: ' + str(div_r_sq))
    plt.title('% of complete shufflings against $d_{div}$ (' + method_name + ')')
    plt.legend()
    plt.xlabel('$d_{div}$')
    plt.ylabel('% of complete shufflings')
    plt.savefig(method_name.replace(' ', '_') + '_d_div_hits.png')
    plt.close()
    # Ranking against d_max
    plt.xlim([-.05, 1.05])
    plt.ylim([-5, 105])
    # print(d_max_hits)
    first_max = d_max_hits.index(max(d_max_hits)) + 1
    trunc_npx = np.array(x[:first_max]).reshape((-1,1))
    npmax = np.array(d_max_hits[:first_max])
    max_model = LinearRegression().fit(trunc_npx, npmax)
    max_r_sq = max_model.score(trunc_npx, npmax)
    max_intercept = max_model.intercept_
    max_slope = max_model.coef_
    xs = [i / 100 for i in range(101)]
    ys = [min(max_intercept + max_slope * xi, 100) for xi in xs]
    plt.grid()
    plt.plot(xs, ys, 'r-', label='Intercept: ' + str(max_intercept) + '\nSlope: ' + str(max_slope[0]) + '\n$R^2$: ' + str(max_r_sq))
    plt.plot(x, d_max_hits, 'o')
    plt.title('% of complete shufflings against $d_{max}$ (' + method_name + ')')
    plt.xlabel('$d_{max}$')
    plt.ylabel('% of complete shufflings')
    plt.legend()
    plt.savefig(method_name.lower().replace(' ', '_') + '_d_max_hits.png')
    plt.close()

def ranking_distance_plotter(results, method_name): #TODO Why here how plot some max distances?
    x = [i / 20 for i in range(21)]
    avg_mu_rank_dist = [0.0] * 21
    max_distance = [0.0] * 21
    min_distance = [1.0] * 21
    total_mus = [0] * 21
    for experiment in results:
        for sample in experiment:
            crisp_ranking = sample['crisp_ranking']
            mu_ranking = sample['mu_ranking']
            if mu_ranking != None:
                print('Found a non-None at: ', sample['d_max'])
                distance = normalized_ell_1(crisp_ranking, mu_ranking)
                avg_mu_rank_dist[int(sample['d_max'] * 20)] += distance
                if distance > max_distance[int(sample['d_max'] * 20)]:
                    max_distance[int(sample['d_max'] * 20)] = distance
                if distance < min_distance[int(sample['d_max'] * 20)]:
                    min_distance[int(sample['d_max'] * 20)] = distance
                total_mus[int(sample['d_max'] * 20)] += 1
    avg_mu_rank_dist = [avg_mu_rank_dist[i] / total_mus[i] for i in range(21)]
    plt.grid()
    plt.plot(x, avg_mu_rank_dist, 'rx-', label='Average $\ell_1-$distance of $\mu Rank$ and $r$')
    plt.fill_between(x, max_distance, min_distance, color=(0.9, 0.9, 0.9), label='Min-max $\ell_1-$distance interval')
    plt.plot([0, 1], [0, 1], 'g-', label='$\ell_1-$distance=$d_{max}$')
    plt.legend()
    plt.xlabel('$d_{max}$')
    plt.ylabel('$\ell_1-$distance')
    plt.title('Average $\ell_1-$distance between shuffled ranking, $\mu Rank$,\nand unshuffled ranking, $r$ (' + method_name + ')')
    plt.savefig(method_name.lower().replace(' ', '_') + '_ranking_distance.png')
    plt.close()

"""
diversity_loss_plotter:
Plots the diversity loss of mu_rank compaired to the desired diversity levels. That is, aggregating over all d_max values for a certain d_div value, 
calculate the corresponding diversity of each initial part of mu_rank and plot it --- you will need to generate 21 plots (?)
x axis: number of entities included --- possibly truncate to minimum number of entities or exclude rankings with size < min_size?
y axis: normalized diversity.
"""

def calculate_distribution(ranking, classes):
    n = int(max(list(classes.values()))) + 1
    distribution = [0] * n
    for point in ranking:
        distribution[classes[str(point[0]) + ',' + str(point[1])]] +=1
    return distribution

def diversity_loss_plotter(results, method_name, diversity_metric, data):
    total = 0
    desired_length = 15
    avg_ranking_loss = [0.0] * desired_length
    for experiment in results:
        for i in range(len(experiment)):
            sample = experiment[i]
            data_sample = data[i]
            mu_rank = sample['mu_ranking']
            classes = data_sample['classes']
            # print(len(mu_rank))
            # print('d_div: ', len(sample['d_div']))
            if mu_rank == None or len(mu_rank) < desired_length:
                continue
            total += 1
            for i in range(desired_length):
                avg_ranking_loss[i] += diversity_metric(calculate_distribution(mu_rank[:i + 1], classes)) - sample['d_div'][i]
    avg_ranking_loss = [x / total for x in avg_ranking_loss]
    x = [i for i in range(1, 16)]
    plt.grid()
    plt.plot(x, avg_ranking_loss, 'bo-', label='Average Diversity Loss')
    plt.title('Average Diversity Loss across all experiments\nagainst length of $\mu Rank$\'s initial part (' + method_name + ')')
    plt.xlabel('Length of $\mu Rank$\' initial part')
    plt.ylabel('Average Diversity Loss')
    plt.xticks(ticks = x)
    # plt.yticks(ticks = [-0.5 + i / 20 for i in range(10)])
    plt.legend()
    plt.savefig(method_name.lower().replace(' ','_') + '_avg_div_loss.png')
    plt.close()

if __name__ == '__main__':
    # with open('shannon_index_results_small.json', 'r') as file:
    #     results = json.load(file)
    methods = ['Richness', 'Shannon Index', 'Berger Parker Index', 'Hill numbers', 'Simpson Index']
    method_functions = [diversity_metrics.richness, diversity_metrics.shannon_index, diversity_metrics.berger_parker_index, diversity_metrics.hill_numbers, diversity_metrics.simpson_index]
    with open('small_data/dataset.json', 'r') as file:
        data = json.load(file)
    # methods = ['Shannon Index']
    for i in range(len(methods)):
        method = methods[i]
        tag = method.lower().replace(' ', '_')
        data_path = tag + '_results_small_2.json'
        with open(data_path, 'r') as file:
            results = json.load(file)
        # successful_rankings_plotter(results, method)
        # successful_rankings_3d_plotter(results, method)
        ranking_distance_plotter(results, method)
        diversity_loss_plotter(results, method, method_functions[i], data)