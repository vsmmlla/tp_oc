#!/usr/bin/python3
import time
import random
import copy
from math import sqrt, inf
import matplotlib.pyplot as plt

from generic import base_problem, greedy, random_greedy, monte_carlo, tabu, simulated_annealing, name_and_time


class ProblemVC(base_problem):

    def __init__(self, n):
        self.nb_cities = n
        self.cities = [[random.random(), random.random()] for _ in range(n)]


    def plot_solution(self, s, title="", color=''):
        X = [self.cities[i][0] for i in s]
        Y = [self.cities[i][1] for i in s]
        X.append(X[0])
        Y.append(Y[0])
        fmt = 'o--'
        if color:
            plt.plot(X, Y, fmt, color=color)
        else:
            plt.plot(X, Y, fmt)
        if title:
            plt.title(title, y=-0.17)


    def random_solution(self):
        res = [i for i in range(self.nb_cities)]
        random.shuffle(res)
        return res


    @name_and_time('naive solution')
    def naive_solution(self):
        def nearest_city(city, list_of_cities):
            shorter_dist = inf
            for tmp_city in list_of_cities:
                tmp_dist = self.dist(self.cities[city], self.cities[tmp_city])
                if tmp_dist < shorter_dist:
                    res = tmp_city
                    shorter_dist = tmp_dist
            return res

        path = []
        remaining = [city for city in range(self.nb_cities)]
        city = random.randint(0, self.nb_cities - 1)
        path.append(city)
        remaining.remove(city)
        while remaining:
            city = nearest_city(city, remaining)
            path.append(city)
            remaining.remove(city)
        return path


    def random_neighbor(self, s):
        res = copy.deepcopy(s)
        i, j = random.randint(0, self.nb_cities - 1), random.randint(0, self.nb_cities - 1)
        res[i], res[j] = res[j], res[i]
        return res


    def all_neighbors(self, s):
        neighbor = copy.deepcopy(s)
        yield neighbor
        for i in range(self.nb_cities):
            for j in range(i + 1, self.nb_cities):
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                yield neighbor
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]


    def dist(self, c1, c2):
        sqdx = (c1[0] - c2[0]) ** 2
        sqdy = (c1[1] - c2[1]) ** 2
        return sqrt(sqdx + sqdy)


    def cost(self, s):
        res = 0
        c_prev = s[-1]
        for c_next in s:
            res += self.dist(self.cities[c_next], self.cities[c_prev])
            c_prev = c_next
        return res


if __name__ == "__main__":

    n       = 200
    nb_iter = 7000

    pb = ProblemVC(n)
    s = pb.random_solution()

    naive = pb.naive_solution()
    h_naive = [pb.cost(naive)] * nb_iter

    start = time.time()
    best_rg, h_best_rg = random_greedy(pb, s, nb_iter)
    best_sa, h_best_sa = simulated_annealing(pb, s, nb_iter, n / 120, 10 / n)
    end = time.time()

    ### Plots ###
    plt.subplot(2, 2, 1)
    plt.title('TSM, {} cities, {} iterations per method, computation : {:.2f}s'.format(n, nb_iter, end - start), x=1.1, fontsize=14)
    plt.plot(h_naive)
    plt.plot(h_best_rg )
    plt.plot(h_best_sa)
    plt.legend(['naive', 'random greedy', 'simulated annealing'])
    plt.gca().set_prop_cycle(None)
    plt.subplot(2, 2, 2)
    pb.plot_solution(naive, title='naive {0:.3f}'.format(pb.cost(naive)), color='tab:blue')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    pb.plot_solution(best_rg, title='random greedy {0:.3f}'.format(pb.cost(best_rg)), color='tab:orange')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 4)
    pb.plot_solution(best_sa, title='simuated annealing {0:.3f}'.format(pb.cost(best_sa)), color='tab:green')
    plt.xticks([])
    plt.yticks([])
    plt.show()
