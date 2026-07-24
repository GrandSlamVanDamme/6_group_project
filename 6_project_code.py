import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

RNG = np.random.default_rng()  # returns random n ϵ

x_LIST = [np.arange(0, 21), np.arange(0, 31), np.arange(0, 41)]
N = 200  # 200 bulbs per batch
k_LIST = [2, 4, 6]  # 1%, 2%, 3% proportion of bad bulbs in a batch
n_LIST = [20, 30, 40]  # bulbs per test sample

# We'll assume a missed bad bulb costs the company 100X that of testing a single bulb.

COST_PER_BULB_TESTED = 0.20  # 20 cents chosen arbitrarily
COST_PER_BAD_BULB_MISSED = (
    100 * COST_PER_BULB_TESTED
)  # speculating based on word of mouth and loss of business over time.


def hypergeo_info(x_LIST, N, k_LIST, n_LIST):
    """
    x = list of random vars x1, x2, ... xn where each xi is a range from 0 to ni
    N = batch/population
    k = list of numbers of successes per batch for different error proportions
    n = list of sample sizes
    """

    x = x_LIST
    k = k_LIST
    n = n_LIST

    for ki in k:
        print()
        print(f"########## With {ki} defective bulbs in the batch ##########")
        print()
        for xi, ni in zip(x, n):
            hg = sci.stats.hypergeom(N, ki, ni)
            P = hg.pmf(xi)
            Ex = hg.mean()
            Vx = hg.var()

            # P_3_or_more_found
            # P_2_or_more_found
            # P_1_or_more_found

            print()
            print(f"Testing {ni} bulbs from the batch of 200 gives us:")
            print(f"Mean number of defective bulbs per sample = {Ex}")
            print(f"Reported variance of {Vx:.2f} per sample")
            print(f"P(3 or more defective bulbs in sample)={1 - hg.cdf(3):.2%}")
            print(f"P(2 or more defective bulbs in sample)={1 - hg.cdf(2):.2%}")
            print(f"P(1 or more defective bulbs in sample)={1 - hg.cdf(1):.2%}")
            print(f"P(No defective bulbs in sample) = {hg.cdf(1):.2%}")
            print()

    return [P, Ex, Vx]


hypergeo_info(x_LIST, N, k_LIST, n_LIST)


def HyperData(N, n, k, T):
    """
    returns data from T trials of a
    hypergeometric distribution with parameters:

    N (batch/population)
    n (sample)
    k (number of target events in batch/population)
    N-k (number of non-target events in batch/population)

    """

    hypgeo = RNG.hypergeometric(k, N - k, n, T)
    count = np.unique_counts(hypgeo)
    return [hypgeo, count]


# print(HyperData(200, 30, 2, 1000))


def histoplot(n, T):

    plt.hist(HyperData(N, n, 2, T)[0], bins=np.linspace(0, 6), alpha=0.5)
    plt.hist(HyperData(N, n, 4, T)[0], bins=np.linspace(0, 6), alpha=0.5)
    plt.hist(HyperData(N, n, 6, T)[0], bins=np.linspace(0, 6), alpha=0.5)
    plt.xlabel(f"number of defective bulbs found in sample of {n} bulbs")
    plt.ylabel("frequency of result over 1000 trials")
    plt.legend(["defect rate = 1%", "defect rate = 2%", "defect rate = 3%"])
    plt.savefig(f"{n}_hist.jpg")
