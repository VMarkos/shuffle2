# Shuffle2
Description of and central results regarding _Shuffle2_ shuffling algorithm.

# Introduction
_Shuffle2_ is a shuffling algorithm which allows for real-valued bounded metrics to be monitored in ranking tasks. More precisely, given a set (to be ranked) _X_, a ranking function, _r_, a metric _m_ which is assumed to take values in _\[0,1\]_, a set _D<sub>m</sub>_ of desired values for that metric and a deviation threshold, _d<sub>max</sub>_, _Shuffle2_ computes a ranking _r*(X)_ by shuffling _r(X)_ such that:

**P1.** _r*(X)_ Has distance at most _d<sub>max</sub>_ from _r(X)_;

**P2.** Each initial part of length _i_ of _r*(X)_ has the closest possible value of _m_ to _D<sub>m</sub>_ given property P1 (see above), for _i=1,2,..._

In the above, the distance between two rankings - i.e., two permutations of the same set - is defined to be their _l1_ distance if each permutation is seen as a sequence of integers. More accurately, we assume that the above metric is _normalized_ - i.e., its values lie on _\[0,1\]_.

The algorithm itself in pseudocode is the following one:

```python
def shuffle2(X, r, m, d_m, d_max):
    crisp_ranking = r(X)
    mu_rank = []
    front = sort_by(m, d_m, X, mu_rank)
    while (len(front) > 0 and len(X) > len(mu_rank)):
        next_el = front.pop()
        mu_rank.append(next_el)
        if normalized_ell_1(crisp_ranking, mu_rank) > d_max:
            mu_rank.pop()
        else:
            front += sort_by(m, d_m, X, mu_rank)
    if len(X) == len(mu_rank):
        return mu_rank
    return crisp_ranking
```
Observe that _Shuffle2_ does not return a shuffled ranking of all the entities on all occasions. This is an expected behaviour since there might be values of d_max that do not allow for any deviations from the initial ranking, in which cases the initial ranking itself is returned.

In fact, _Shuffle2_ is a backtracking search with the backtracking condition being that at any time, the shuffled ranking should not deviated from _r(X)_ more than _d<sub>max</sub>_. Hence, while _d<sub>max</sub>_ provides a **strict** restriction for _Shuffle2_, the condition _m(r*<sub>i</sub>)_ to belong to _d_m_ for most initial parts _r*<sub>i</sub>_ of the shuffled ranking _r*_ is a **soft** restriction, meaning that _Shuffle2_ does not commit to it in cases where _d<sub>max</sub>_ is too restrictive.

# The special case of diversity
In order to study _Shuffle2_ we will utilize diversity-related metrics - thus, _m_ and _div_ for a metric will be used interchangeably.

## Some diversity metrics
In short, we will make use of the following diversity metrics, used in several fields, ranging from Ecosystem Biology to Economics and HR management:
* **Richness**, _R_, which is defined as the number of different classes (e.g., species) that are present in a certain set.
* **Shannon's Index**, _H_, which is exactly the information theoretic _entropy_, i.e.:

    ![equation](https://latex.codecogs.com/gif.latex?H%3D-%5Csum_%7Bk%3D1%7D%5Enp_k%5Cln%20p_k)

    In the above equation, _p<sub>i</sub>_ denote the relative frequencies of each class in the dataset.
* **Simpson's Index**, _λ_, which is roughly defined as the probability that two entities belonging to the same class are picked consecutively., i.e.:

    ![equation](https://latex.codecogs.com/gif.latex?%5Clambda%3D%5Csum_%7Bk%3D1%7D%5Enp_k%5E2)
* **Berger-Parker Index**, _BP_, which is the maximum relative frequency observed in the dataset.
* **Hill Numbers**, _<sup>q</sup>D_, which correspond roughly to the number of equally abundant classes that would be required in order to achieve the same average proportional abundance in our dataset - i.e., how many classes with the same number of members we need in order to match the average number of members per class in an existing dataset. Hill numbers are defined as follows, where _q&GreaterEqual;1_

    ![equation](https://latex.codecogs.com/gif.latex?%5EqD%3D%5Cleft%28%5Csum_%7Bk%3D1%7D%5Enp_i%5Eq%5Cright%29%5E%7B1/%281-q%29%7D)
    
## Empirical evaluation of _Shuffle2_
Using the above metrics we measured several parameters regarding _Shuffle2_. For our purposes, we generated a small dataset consisting of 10 samples, with each sample containing four (4) clusters of points in _[0,1]<sup>2</sup>_ and each cluster containing 0 to 10 points. For each sample, we also determined a point, _x_, which served as the target point for our ranking. Regarding ranking, we ranked all points in each sample separately with respect to their euclidean distance from _x_ from the closest to the farthest. Then we proceeded as follows:

```python
for div_metric in [richness, shannon_index, berger_parker_index, simpson_index, hill_numbers]:
    for sample in dataset:
        for d_div in [0.0, 0.05, ..., 1.0]:
            for d_max in [0.0, 0.05, ..., 1.0]:
                mu_rank = shuffle2(sample, r, div_metric, d_div, d_max)
```

That is, for all the diversity metrics we described above, we run _Shuffle2_ in all the samples of our dataset for various values of desired diversity, _d<sub>div</sub>_ and for various values of maximum deviation, _d<sub>max</sub>_, from our initial ranking, _r_ - in the above, _r_ is the euclidean distance ranking we discussed above.

The dataset as well as the results may be found [here](https://github.com/VMarkos/shuffle2/tree/main/small_data) and [here](https://github.com/VMarkos/shuffle2/tree/main/small_results) respectively.

## Results
As we mentioned above, _Shuffle2_ may not always succeed in shuffling the given ranking since there might not be any space for improvement - e.g., when _d<sub>max</sub> = 0_ or, in general, _d<sub>max</sub> &thickapprox; 0_. So, it waould be meaningful to study how often _Shuffle2_ manages to shuffle a given ranking with respect to _d_<sub>max</sub>_ and _d<sub>div</sub>_. The images below show the percentage of shuffled rankings against both parameters:

| Results | Results |
| --- | --- |
![Richness](https://github.com/VMarkos/shuffle2/blob/main/small_results/Richness_3d_hits.png?raw=true) | ![Shannon's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/Shannon_Index_3d_hits.png?raw=true)
![Simpson's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/Simpson_Index_3d_hits.png?raw=true) | ![Berger-Parker Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/Berger_Parker_Index_3d_hits.png?raw=true)
![Hill Numbers](https://github.com/VMarkos/shuffle2/blob/main/small_results/Hill_numbers_3d_hits.png?raw=true) | 

As one may suspect from the above, in none of the five metrics we have examined _d<sub>div</sub>_ plays any important role in the number of shuffled rankings - since, as we have discussed, it serves only as a **soft** restriction. Indeed, this is verified in the plots below:

| Results | Results |
| --- | --- |
![Richness](https://github.com/VMarkos/shuffle2/blob/main/small_results/Richness_d_div_hits.png?raw=true) | ![Shannon's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/Shannon_Index_d_div_hits.png?raw=true)
![Simpson's INdex](https://github.com/VMarkos/shuffle2/blob/main/small_results/Simpson_Index_d_div_hits.png?raw=true) | ![Berger-Parker Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/Berger_Parker_Index_d_div_hits.png?raw=true)
![Hill Numbers](https://github.com/VMarkos/shuffle2/blob/main/small_results/Hill_numbers_d_div_hits.png?raw=true) | 

Given the above, we aggregate the above results for each value of _d<sub>max</sub>_ and all values of _d<sub>div</sub>_ and get the following results - square residual error, _R<sup>2</sup>_ refers only to the non-constant part of the curve, since including the ending constant part in our computations would result to more optimistic results regarding _R<sup>2</sup>_.

| Results | Results |
| --- | --- |
![Richness](https://github.com/VMarkos/shuffle2/blob/main/small_results/richness_d_max_hits.png?raw=true) | ![Shannon's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/shannon_index_d_max_hits.png?raw=true)
![Simpson's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/simpson_index_d_max_hits.png?raw=true) | ![Berger-Parker Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/berger_parker_index_d_max_hits.png?raw=true)
![Hill Numbers](https://github.com/VMarkos/shuffle2/blob/main/small_results/hill_numbers_d_max_hits.png?raw=true) | 

In general, the above results agree to our initial intuition that smaller values of _d<sub>max</sub>_ would lead to less shufflings than larger ones, which is a natural consequence of _d<sub>max</sub>_ being a **strict** restriction. Next, we study how the shuffled rankings returned by _Shuffle2_ differ with respect to the initial ranking, _r(X)_. As shown below, for lower values of _d<sub>max</sub_, _Shuffle2_ returns shuffled rankings that in general have an average distance of _d<sub>max</sub_ from the initial ranking - which hints to a desired behaviour, i.e., that _Shuffle2_ returns a ranking as far as possible from our initial one in order to introduce more diversity in the shuffled ranking. However, for lagrger values of _d<sub>max</sub_, this distance seems to converge to a fixed value. This is also expected in the sense that, on average, once enough entities of _X_ have been ranked, the introduction of a new entity does not impact the (shuffled) ranking diversity as much.

| Results | Results |
| --- | --- |
![Richness](https://github.com/VMarkos/shuffle2/blob/main/small_results/richness_ranking_distance.png?raw=true) | ![Shannon's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/shannon_index_ranking_distance.png?raw=true)
![Simpson's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/simpson_index_ranking_distance.png?raw=true) | ![Berger-Parker Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/berger_parker_index_ranking_distance.png?raw=true)
![Hill Numbers](https://github.com/VMarkos/shuffle2/blob/main/small_results/hill_numbers_ranking_distance.png?raw=true) | 
    
At last, we also studied the amount of diversity sacrificed/gained with respect to each metric compaired to the desired levels of diversity, _d<sub>div</sub>_.

| Results | Results |
| --- | --- |
![Richness](https://github.com/VMarkos/shuffle2/blob/main/small_results/richness_avg_div_loss.png?raw=true) | ![Shannon's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/shannon_index_avg_div_loss.png?raw=true)
![Simpson's Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/simpson_index_avg_div_loss.png?raw=true) | ![Berger-Parker Index](https://github.com/VMarkos/shuffle2/blob/main/small_results/berger_parker_index_avg_div_loss.png?raw=true)
![Hill numbers](https://github.com/VMarkos/shuffle2/blob/main/small_results/hill_numbers_avg_div_loss.png?raw=true) | 

At first, we should point out that negative values of Average Diversity Loss are interpreted as diversity deficit compared to _d<sub>div</sub>_ while positive values as diverstity surplus. This is, as one easily observes, the only factor in which the five diversity metrics we have discussed seem to have significant differences. For instance, Richness seems to be increasing with the number of entries introduced in the ranking (on average) while Shannon's Index seems to be intially sharply increasing and the to converge to some certain value - roughly about _-0.1_. Such behaviour should, in the first place, be attributed to the way each metric measures diversity. For instance, Richness measures the numberof classes present in our dataset and, the normalized version we have used in our experiments, measures the number of classes present over the total number of classes that could be presnt in a sample. Consequently, since any diversity restrictions are **soft**, as we integrate more members of _X_ in our ranking, we expect (normalized) Richness to rise. Similarly, one may also explain the behaviour of the remaining four (4) diversity metrics.
