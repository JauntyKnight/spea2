"""
SPEA2 algorithm implementation for multi-objective optimization

https://doi.org/10.3929/ethz-a-004284029
"""

import numpy as np
import math

from random import sample
from copy import deepcopy
from multiprocess import Pool
from sklearn.neighbors import KDTree


# Wrapper for a custom individual class
# it only adds the objectives and fitness attributes
class IndividualWrapper:
    def __init__(self, individual):
        self.individual = individual
        self.objectives = None
        self.fitness = None

    def __str__(self):
        return str(self.individual)

    def __repr__(self):
        return repr(self.individual)

    def mutate(self):
        return IndividualWrapper(self.individual.mutate())


def euclidean_distance(x, y):
    """
    Compute the Euclidean distance between two points in n-dimensional space
    """
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) ** 0.5


def argmin(iterable):
    """
    Return the index of the minimum element in the iterable
    """
    return min(enumerate(iterable), key=lambda x: x[1])[0]


def tournament_selection(population, tournament_size):
    """
    Select the best individual from a tournament
    """
    tournament = sample(population, tournament_size)
    winner = min(tournament, key=lambda individual: individual.fitness)
    return deepcopy(winner)


def compute_fitnesses(popuation, dominates):
    """
    Computes the fitness of each individual in the population (as required by SPEA2)

    The fitness of an individual is the sum of the strengths of the individuals that dominate it
    plus the density of the individual in the objective space

    The strength of an individual is the number of individuals that it dominates
    The density of an individual is the inverse of the distance to the k-th nearest neighbor

    Args:
        popuation (list): list of IndividualWrapper objects
        dominates (function): function that returns True if the first objective tuple dominates the second objective tuple
    """
    strengths = [
        sum(
            dominates(individual.objectives, other.objectives)
            for other in popuation
        )
        for individual in popuation
    ]

    for individual in popuation:
        individual.fitness = sum(
            strength_other
            for other, strength_other in zip(popuation, strengths)
            if dominates(other.objectives, individual.objectives)
        )

    k = int(len(popuation) ** 0.5) + 1  # neighborhood size

    # transform the population to a matrix
    population_matrix = np.array(
        [individual.objectives for individual in popuation]
    )

    # create a KDTree
    tree = KDTree(population_matrix)

    densities = 1 / (tree.query(population_matrix, k=k)[0][:, -1] + 2)

    for individual, density in zip(popuation, densities):
        individual.fitness += density

    return popuation


def compute_new_archive(population, archive_size):
    """
    Gets the `archive_size` best individuals from the population

    If the population is smaller than the archive size, the best dominated individuals are added to fit the archive size
    If the population is larger than the archive size, the least diverse individuals are removed

    Args:
        population (list): list of IndividualWrapper objects
        archive_size (int): size of the archive
    """
    new_archive = [
        individual for individual in population if individual.fitness < 1
    ]

    if len(new_archive) < archive_size:
        # Add the best dominated individuals to fit the archive size
        new_archive += sorted(
            filter(lambda x: x.fitness >= 1, population),
            key=lambda x: x.fitness,
        )[: archive_size - len(new_archive)]

    elif len(new_archive) > archive_size:
        # remove the excessive individuals that contribute the least
        # to the deiversity of the objectives

        # create a matrix of distances between the objectives
        distances = [
            [
                euclidean_distance(individual.objectives, other.objectives)
                for other in new_archive
            ]
            for individual in new_archive
        ]

        to_remove = set()

        while True:
            # get the individual that is the closest to the others
            closest_individual = argmin(map(sorted, distances))

            to_remove.add(closest_individual)

            if len(to_remove) == len(new_archive) - archive_size:
                break

            for i in range(len(distances)):
                distances[i][closest_individual] = math.inf
                distances[closest_individual][i] = math.inf

        new_archive = [
            individual
            for i, individual in enumerate(new_archive)
            if i not in to_remove
        ]

    return new_archive


def spea2(
    Individual,
    evaluation_fun,
    dominates,
    population_size,
    archive_size,
    max_generations,
    verbose=True,
    n_jobs=1,
    callback=None,
):
    """
    SPEA2 algorithm for multi-objective optimization (https://doi.org/10.3929/ethz-a-004284029)
    The algorithm is a genetic algorithm that uses a combination of the population and an archive to find the Pareto front

    Args:
        Individual (class): class of the candidate solutions
        evaluation_fun (function): function that takes an individual and returns a tuple of objectives
        dominates (function): function that returns True if the first objective tuple dominates the second objective tuple
        population_size (int): size of the population
        archive_size (int): size of the archive
        max_generations (int): maximum number of generations
        verbose (bool): if True, print the progress of the algorithm
        n_jobs (int): number of parallel jobs
        callback (function): function that is called at the end of each generation
    """
    # Create the initial population and the archive
    population = [
        IndividualWrapper(Individual()) for _ in range(population_size)
    ]
    archive = []

    all_time_population = set()

    # Main loop
    for generation in range(max_generations):
        with Pool(n_jobs) as pool:
            # Compute the fitnesses of the population
            objectives_population = pool.map(
                lambda x: evaluation_fun(x.individual), population
            )

        for individual, objectives in zip(population, objectives_population):
            individual.objectives = objectives

        # Combine the population and the archive
        combined_population = population + archive

        # Compute the fitnesses of the population
        compute_fitnesses(combined_population, dominates)

        all_time_population.update(combined_population)

        # Update the archive
        archive = compute_new_archive(combined_population, archive_size)

        if verbose:
            print(f"Generation {generation + 1}/{max_generations}")
            for i in range(len(archive[0].objectives)):
                print(f"Objective {i + 1}:")
                print(f"  Best: {max(archive, key=lambda x: x.objectives[i])}")
                print(f"  Worst: {min(archive, key=lambda x: x.objectives[i])}")
                print()

        # Callback
        if callback is not None:
            callback(generation, population, archive)

        # Select and mutate the next population
        population = [
            tournament_selection(archive, 2).mutate()
            for _ in range(population_size)
        ]

    return population, archive
