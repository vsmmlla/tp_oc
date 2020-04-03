import copy
import time
import random
from math import inf, exp

import matplotlib.pyplot as plt


class base_problem:

    def cost(self, s):
        pass


    def random_neighbor(self, s):
        pass


    def all_neighbors(self, s):
        pass


    def random_solution(self):
        pass


    def neigh_argmin_cost(self, s, tabu=[]):
        res = copy.deepcopy(s)
        cost = self.cost(res)
        all_neighbors = self.all_neighbors(res)
        nb_step = 0
        for neighbor in all_neighbors:
            if not neighbor in tabu:
                nb_step += 1
                tmp = self.cost(neighbor)
                if tmp < cost:
                    cost = tmp
                    res = copy.deepcopy(neighbor)
        return res, nb_step


def name_and_time(method_name):
    def decorator(method):
        def wrapper(*args, **kargs):
            print("***", method_name, "start ***")
            start = time.time()
            res = method(*args, **kargs)
            end = time.time()
            print("***", method_name, "end : Computation took {0:.2f} seconds ***".format(end - start))
            return res
        return wrapper
    return decorator


@name_and_time('Monte-Carlo')
def monte_carlo(pb, iter_max):
    history = []
    res = pb.random_solution()
    best_cost = pb.cost(res)
    print(best_cost, end=" ", flush=True)
    for _ in range(iter_max - 1):
        tmp = pb.random_solution()
        tmp_cost = pb.cost(tmp)
        if tmp_cost < best_cost:
            best_cost = tmp_cost
            res = tmp
            print(best_cost, end=" ", flush=True)
        history.append(best_cost)
    print()
    return res, history


@name_and_time('Greedy')
def greedy(pb, s):
    history = []
    res = copy.deepcopy(s)
    best_cost = inf
    tmp_cost = pb.cost(res)
    try:
        while tmp_cost < best_cost:
            best_cost = tmp_cost
            res, nb_steps = pb.neigh_argmin_cost(res)
            history = history + (nb_steps * [best_cost])
            tmp_cost = pb.cost(res)
            print(best_cost, end=" ", flush=True)
    except KeyboardInterrupt:
        pass
    else:
        raise
    print()
    return res, history


@name_and_time('Random greedy')
def random_greedy(pb, s, iter_max):
    history = []
    res = copy.deepcopy(s)
    best_cost = pb.cost(s)
    print(best_cost, end=" ", flush=True)
    last_cost = 2 * pb.cost(s)
    for _ in range(iter_max):
        tmp = pb.random_neighbor(res)
        tmp_cost = pb.cost(tmp)
        if tmp_cost < best_cost:
            best_cost = tmp_cost
            res = tmp
            print(best_cost, end=" ", flush=True)
        history.append(best_cost)
    print()
    return res, history
    

@name_and_time('Tabu')
def tabu(pb, s, iter_max, tabou_len_max):
    history = []
    best_cost = inf
    tabu = []
    res = copy.deepcopy(s)
    tmp = res
    tmp_cost = pb.cost(tmp)
    for _ in range(iter_max):
        if tmp_cost < best_cost:
            best_cost = tmp_cost
            res = tmp
            print(tmp_cost, end=" ", flush=True)
        if len(tabu) == tabou_len_max:
            tabu.pop(0)
        tabu.append(tmp)
        tmp, nb_steps = pb.neigh_argmin_cost(tmp, tabu=tabu)
        history = history + (nb_steps * [best_cost])
        tmp_cost = pb.cost(tmp)
    print()
    return res, history


@name_and_time('Simulated annealing')
def simulated_annealing(pb, s, iter_max, beta_0, eps):
    history = []
    beta = beta_0
    res = copy.deepcopy(s)
    current_cost = pb.cost(res)
    accepted = False
    nb_steps = 0
    for _ in range(iter_max):
        nb_steps += 1
        tmp = pb.random_neighbor(res)
        tmp_cost = pb.cost(tmp)
        if tmp_cost < current_cost:
            accepted = True
        else:
            p = exp(-beta * (tmp_cost - current_cost))
            x = random.random()
            if x < p:
                accepted = True
        if accepted:
            res = tmp
            history = history + (nb_steps * [current_cost])
            nb_steps = 0
            print(tmp_cost, end=" ", flush=True)
            current_cost = tmp_cost
            accepted = False
        beta = (1 + eps) * beta
    print()
    return res, history
