import numpy as np
import matplotlib.pyplot as plt
import scipy as sci


def hypergeo_info(x, N, k, n):

    for ni in n:
        for xi, ki in zip(x, k):
            P = sci.stats.hypergeom.pmf(xi, N, ki, ni)
            Ex = sci.stats.hypergeom.mean(xi, N, ki, ni)
            Vx = sci.stats.hypergeom.var(xi, N, ki, ni)
            print(
                f"For a sample of {ni} bulbs, given {ki} defective bulbs in the batch, we report an expected value of {Ex} bulbs and a variance of {Vx}."
            )

    return [P, Ex, Vx]


def HyperData(N, n, k, T):
    """
    returns data from T trials of a
    hypergeometric distribution with parameters:

    N (batch/population)
    n (sample)
    k (number of target events in batch/population)
    N-k (number of non-target events in batch/population)

    """

    total = []
    counter = 0

    while counter < T:
        counter += 1

        total.append(
            np.random.Generator.hypergeometric(k, N - k, n)
        )  # returns draws from sample

        # print(f'''There were {hypgeo} successful draws from trial {counter}.''')

    return total


print(HyperData(200, 30, 2, 1000))
