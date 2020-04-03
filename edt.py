#!/usr/bin/python3
import copy
import random

import matplotlib.pyplot as plt

from generic import base_problem, greedy, random_greedy, monte_carlo, tabu, simulated_annealing


class ProblemEDT(base_problem):

    def __init__(
            self,
            nb_slots, # nombre de craineaux
            nb_profs, # nombre de profs
            nb_groups, # nombre de classes
            nb_group_per_prof, # nombre de classes suivi par un prof
            nb_prof_per_group, # nombre de prof par classes
            nb_course_per_prof): # nombre de cours donne par un prof dans une classe
        
        self.nb_slots = nb_slots
        self.nb_profs = nb_profs
        self.nb_groups = nb_groups

        if nb_profs * nb_group_per_prof != nb_groups * nb_prof_per_group:
            raise ValueError("Not compatible arguments")

        self.nb_group_per_prof = nb_group_per_prof
        self.nb_prof_per_group = nb_prof_per_group
        self.prof_allocation = self.allocate(self.nb_profs, self.nb_groups,
                                   nb_group_per_prof, nb_prof_per_group)

        nb_empty_slots = nb_slots - nb_course_per_prof * nb_prof_per_group
        if nb_empty_slots < 0:
            raise ValueError("Not enough slots.")

        self.nb_course_per_prof = nb_course_per_prof


    def allocate(self, nb_profs, nb_groups,
            nb_group_per_prof, nb_prof_per_group):

        to_be_affected = []
        for prof in range(1, nb_profs + 1):
            for _ in range(nb_group_per_prof):
                to_be_affected.append(prof)

        A = [[to_be_affected[ii + i * nb_groups] for i in range(nb_prof_per_group)] for ii in range(nb_groups)]

        return A


    def print_sol(self, s):
        for y in s:
            print('\t'.join([str(x) for x in y]))


    def nb_collisions(self, s):
        collisions = 0
        prof_needed = [0 for _ in range(self.nb_profs + 1)]

        for slot in range(self.nb_slots):

            for group in range(self.nb_groups):
                prof_needed[ s[group][slot] ] += 1

            for i in range(1, self.nb_profs + 1):
                if prof_needed[i] != 0:
                    collisions += prof_needed[i] - 1
                    prof_needed[i] = 0
            prof_needed[0] = 0

        return collisions


    def assert_solution(self, s):
        '''
        Check if s is a solution ie :
            - s is a list of list of int between 0 and self.profs (included)
              (0 mean no class or "etude" i from 1 to self.profs mean prof i)
            - each group receive only course from allocated professor
            - each prof give 3 classes per allocated group
        '''

        professors = [i for i in range(self.nb_profs + 1)]
        assert isinstance(s, list), "The solution must be a list of list of int."
        for y in s:
            assert isinstance(y, list), "The solution must be a list of list of int."
            for x in y:
                assert x in professors, \
                "Professor should be between 0 and {}".format(self.nb_profs)
        for group in range(len(s)):
            count = {prof:0 for prof in self.prof_allocation[group]}
            for slot in range(len(s[0])):
                if s[group][slot] != 0:
                    assert s[group][slot] in self.prof_allocation[group], \
                    "Professor of group {} on slot {} is not in his allocated group.".format(group, slot)
                    count[ s[group][slot] ] += 1
            for prof in self.prof_allocation[group]:
                assert count[prof] == 3, "Professor {} doesn't give three course.".format(prof)


    def cost(self, s):
        colls = self.nb_collisions(s)
        return colls


    def random_solution(self):
        s = [[] for _ in range(self.nb_groups)]
        for group in range(self.nb_groups):
            s[group] = [prof for prof in self.prof_allocation[group]]
            s[group] = self.nb_course_per_prof * s[group]
            nb_empty_slots = self.nb_slots - self.nb_course_per_prof * self.nb_prof_per_group
            s[group] += nb_empty_slots * [0] # 0 mean etude
            random.shuffle(s[group])
        return s
    

    def all_neighbors(self, s):
        tmp = copy.deepcopy(s)
        yield tmp
        for group in range(self.nb_groups):
            for i in range(self.nb_slots):
                for j in range(i + 1, self.nb_slots):
                    tmp[group][i], tmp[group][j] = tmp[group][j], tmp[group][i]
                    yield tmp
                    tmp[group][i], tmp[group][j] = tmp[group][j], tmp[group][i]

    
    def random_neighbor(self, s):
        res = copy.deepcopy(s)
        group = random.randint(0, self.nb_groups - 1)
        i = random.randint(0, self.nb_slots - 1)
        j = random.randint(0, self.nb_slots - 1)
        res[group][i], res[group][j] = res[group][j], res[group][i]
        return res




if __name__ == '__main__':

    K   = 20 # nb slots
    M   = 32 # nb profs
    N   = 16 # nb groups
    ngp = 3  # nombre de classes suivi par un prof
    npg = 6  # nombre de prof par classes
    ncp = 3  # nombre de cours donne par un prof dans une classe
    pb = ProblemEDT(K, M, N, ngp, npg, ncp)

    K   = 40 # nb slots
    M   = 64 # nb profs
    N   = 32 # nb groups
    ngp = 6  # nombre de classes suivi par un prof
    npg = 12 # nombre de prof par classes
    ncp = 3  # nombre de cours donne par un prof dans une classe
    big_pb = ProblemEDT(K, M, N, ngp, npg, ncp)

    s = pb.random_solution()

    #r1, h1 = greedy(pb, s)
    #r2, h2 = tabu(pb, s, 20, 1200)
    r3, h3 = random_greedy(pb, s, 800)
    r4, h4 = simulated_annealing(pb, s, 800, (K * N) / 100, 10 / (K * N))

    #plt.plot(h1)
    #plt.plot(h2)
    plt.plot(h3)
    plt.plot(h4)

    plt.show()


