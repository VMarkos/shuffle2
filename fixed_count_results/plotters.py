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

def diversity_loss_plotter(results, method_name, diversity_metric, data, n_total):
    total = 0
    desired_length = n_total
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
                avg_ranking_loss[i] += abs(diversity_metric(calculate_distribution(mu_rank[:i + 1], classes)) - sample['d_div'][i][0]) #FIXME This is not generic
    avg_ranking_loss = [x / total for x in avg_ranking_loss]
    x = [i for i in range(1, desired_length + 1)]
    plt.grid()
    plt.plot(x, avg_ranking_loss, 'bo-', label='Average Diversity Loss')
    plt.title('Average Diversity Loss across all experiments\nagainst length of $\mu Rank$\'s initial part (' + method_name + ')')
    plt.xlabel('Length of $\mu Rank$\' initial part')
    plt.ylabel('Average Diversity Loss')
    # plt.xticks(ticks = x)
    # plt.yticks(ticks = [-0.5 + i / 20 for i in range(10)])
    plt.legend()
    plt.savefig(method_name.lower().replace(' ','_') + '_avg_div_loss_n_total=' + str(n_total) + '.png')
    plt.close()

def exec_time_plotter(): # You need to take into account all results files for that method!
    results_suffix = 'fixed_count_results'
    methods = ['Richness', 'Shannon Index', 'Berger Parker Index', 'Hill numbers', 'Simpson Index']
    avg_times = {method: [0.0] * 17 for method in methods}
    max_times = {method: [0.0] * 17 for method in methods}
    min_times = {method: [math.inf] * 17 for method in methods}
    xs = [x for x in range(15, 100, 5)]
    for metric_id in methods:
        for n_total in range(15, 100, 5):
            with open('fixed_count_results/' + metric_id.lower().replace(' ', '_') + results_suffix + 'n_total=' + str(n_total) + '.json', 'r') as file:
                results = json.load(file)
            total_results = len(results) * len(results[0])
            for experiment in results:
                for sample in experiment:
                    exec_time = sample['exec_time']
                    pos = int(n_total / 5 - 3)
                    avg_times[metric_id][pos] += exec_time / total_results
                    if exec_time > max_times[metric_id][pos]:
                        max_times[metric_id][pos] = exec_time
                    if exec_time < min_times[metric_id][pos]:
                        min_times[metric_id][pos] = exec_time
        # plt.plot(xs, avg_times[metric_id], 'rx-', label='Average Execution Time')
        # plt.fill_between(xs, min_times[metric_id], max_times[metric_id], color=(0.9, 0.9, 0.9), label='Min-Max Interval')
        # plt.grid()
        # plt.title('Execution Time (' + metric_id + ')')
        # plt.xlabel('# of ranked items')
        # plt.ylabel('Shuffling Time (seconds)')
        # plt.legend()
        # plt.savefig(metric_id.lower().replace(' ','_') + 'exec_time_all_fit.png')
        # plt.close()
    total_max_times = [max([max_times[method][i] for method in methods]) for i in range(17)]
    total_min_times = [min([min_times[method][i] for method in methods]) for i in range(17)]
    avg_time = [0.0] * 17
    for times in avg_times.values():
        for i in range(len(times)):
            avg_time[i] += times[i]
    avg_time = [x / len(methods) for x in avg_time]
    # plt.loglog()
    plt.plot(xs, avg_time, 'bo')
    logx = [math.log10(x) for x in xs]
    logy = [math.log10(y) for y in avg_time]
    npx = np.array(logx).reshape((-1,1))
    npy = np.array(logy)
    lin_model = LinearRegression().fit(npx, npy)
    intercept = lin_model.intercept_
    coef = lin_model.coef_
    r_sq = lin_model.score(npx, npy)
    # plt.plot(xs, [10 ** intercept * math.pow(x, coef) for x in xs], 'r-', label='Intercept=' + str(intercept) + '\nSlope=' + str(coef[0]) + '\n$R^2=$' + str(r_sq))
    plt.plot(xs, [10 ** intercept * math.pow(x, coef) for x in xs], 'r-', label='$y=10^{-6.576}x^{3.032}$')
    plt.fill_between(xs, total_min_times, total_max_times, color=(0.9, 0.9, 0.9), label='Min-Max Interval')
    plt.legend()
    plt.grid()
    plt.title('Execution Time of all Methods')
    plt.xlabel('# of ranked items')
    plt.ylabel('Shuffling Time (seconds)')
    plt.savefig('exec_time_all_fit.png')
    plt.close()
    # Orthonormal plot
    # plt.legend()
    # plt.grid()
    # plt.title('Execution Time of all Methods')
    # plt.xlabel('# of ranked items')
    # plt.ylabel('Shuffling Time (seconds)')
    # plt.savefig('exec_time_all.png')
    # plt.close()
    # LogLog plot:
    # plt.legend()
    # plt.grid()
    # plt.loglog()
    # plt.title('Execution Time of all Methods (log-log)')
    # plt.xlabel('# of ranked items')
    # plt.ylabel('Shuffling Time')
    # plt.savefig('exec_time_all_loglog.png')
    # plt.close()

def get_time_classes(min_time, max_time, times, bin_count):
    # bin_count = 20
    dist = [0] * (bin_count + 1)
    bin_length = (max_time - min_time) / bin_count
    step = 1 / len(times)
    for time in times:
        # print(int((time - min_time) / bin_length))
        dist[int((time - min_time) / bin_length)] += step
    return dist

def exec_time_vs_d_max_plotter():
    results_suffix = 'fixed_count_results'
    methods = ['Richness', 'Shannon Index', 'Berger Parker Index', 'Hill numbers', 'Simpson Index']
    avg_times = [0.0] * 21
    max_times = [0.0] * 21
    min_times = [math.inf] * 21
    times_distribution = [[] for _ in range(21)]
    xs = [x for x in range(21)]
    for metric_id in methods:
        for n_total in range(15, 100, 5):
            with open('fixed_count_results/' + metric_id.lower().replace(' ', '_') + results_suffix + 'n_total=' + str(n_total) + '.json', 'r') as file:
                results = json.load(file)
            total_results = len(results) * len(results[0])
            for experiment in results:
                for sample in experiment:
                    exec_time = sample['exec_time']
                    pos = int(sample['d_max'] * 20)
                    avg_times[pos] += exec_time / total_results
                    times_distribution[pos].append(exec_time)
                    if exec_time > max_times[pos]:
                        max_times[pos] = exec_time
                    if exec_time < min_times[pos]:
                        # print(exec_time)
                        min_times[pos] = exec_time
    print(len(times_distribution))
    plt.plot(xs, avg_times, 'rx-', label='Average Execution Time')
    plt.fill_between(xs, min_times, max_times, color=(0.9, 0.9, 0.9), label='Min-Max Interval')
    # bin_count = 20
    # for i in range(21):
    #     time_dist = get_time_classes(min_times[i], max_times[i], times_distribution[i], bin_count)
    #     for j in range(bin_count + 1):
    #         step = (max_times[i] - min_times[i]) / 20
    #         plt.fill_between([i - 0.5, i + 0.5], [min_times[i] + j * step] * 2, [min_times[i] + (j + 1) * step] * 2, color=(1.0, 0.0, 0.0, 0.5 - 0.8*time_dist[j]))#, time_dist[j], time_dist[j]))
    plt.grid()
    plt.xlabel('$d_{max}$')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time against $d_{max}$')
    plt.savefig('exec_time_vs_d_max_distribution.png')
    plt.close()

if __name__ == '__main__':
    # with open('shannon_index_results_small.json', 'r') as file:
    #     results = json.load(file)
    methods = ['Richness', 'Shannon Index', 'Berger Parker Index', 'Hill numbers', 'Simpson Index']
    method_functions = [diversity_metrics.richness, diversity_metrics.shannon_index, diversity_metrics.berger_parker_index, diversity_metrics.hill_numbers, diversity_metrics.simpson_index]
    exec_time_vs_d_max_plotter()
    # with open('small_data/dataset.json', 'r') as file:
    #     data = json.load(file)
    # methods = ['Shannon Index']
    # exec_time_plotter()
    # for n_total in range(15, 100, 15):
    #     with open('fixed_count_data/dataset_n_total=' + str(n_total) + '.json', 'r') as file:
    #         data = json.load(file)
    #     for i in range(len(methods)):
    #         method = methods[i]
    #         tag = method.lower().replace(' ', '_')
    #         data_path = 'fixed_count_results/' + tag + 'fixed_count_resultsn_total=' + str(n_total) + '.json'
    #         with open(data_path, 'r') as file:
    #             results = json.load(file)
    #         # successful_rankings_plotter(results, method)
    #         # successful_rankings_3d_plotter(results, method)
    #         # ranking_distance_plotter(results, method)
    #         diversity_loss_plotter(results, method, method_functions[i], data, n_total)