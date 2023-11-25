import numpy as np
import time
from scipy.stats import beta

class BernoulliBandit():

    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            np.random.seed(42)
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0


class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        self.eps = eps

        self.estimates = [init_proba] * self.bandit.n  # Optimistic initialization

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.n)
        else:
            # Pick the best one.
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i

class EpsilonDecreasingGreedy(Solver):
    def __init__(self, bandit, eps, decay, init_proba=1.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super(EpsilonDecreasingGreedy, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        self.eps = eps
        self.decay = decay

        self.estimates = [init_proba] * self.bandit.n  # Optimistic initialization

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.n)
        else:
            # Pick the best one.
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        self.eps = self.eps * self.decay
        return i


class UCB1(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class BayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(BayesianUCB, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit.n),
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(
                self._as[x], self._bs[x]) * self.c
        )
        r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i


class ThompsonSampling(Solver):
    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit)

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
    
def experiment(K, N):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.
    Args:
        K (int): number of slot machiens.
        N (int): number of time steps to try.
    """

    b = BernoulliBandit(K)
    print("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
    print("The best machine has index: {} and proba: {}".format(
        max(range(K), key=lambda i: b.probas[i]), max(b.probas)))
    ucb = UCB1(b)
    test_solvers = [
        # EpsilonGreedy(b, 0),
        # EpsilonGreedy(b, 1),
        EpsilonGreedy(b, 0.1),
        ucb,
        BayesianUCB(b, 3, 1, 1),
        ThompsonSampling(b, 1, 1),
        EpsilonDecreasingGreedy(b, 0.1, 0.95)
    ]
    names = [
        # 'Full-exploitation',
        # 'Full-exploration',
        r'$\epsilon$' + '-Greedy',
        'UCB1',
        'Bayesian UCB',
        'Thompson Sampling',
        r'$\epsilon$' + '-Decreasing-Greedy'
    ]

    for s in test_solvers:
        s.run(N)
        print(s.estimated_probas)

    # plot_results(test_solvers, names, "./Project/results_K{}_N{}.png".format(K, N))

    # print(ucb.counts)
    # print(ucb.t)


if __name__ == '__main__':
    experiment(10, 5000)