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
* **Berger-Parker Index**, _BP_, which is
