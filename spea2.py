import numpy as np
import math
import random
import os


from multiprocessing import Pool
from sklearn.neighbors import KDTree


def euclidean_distance(x, y):
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) ** 0.5


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    winner = min(tournament, key=lambda individual: individual.fitness)
    return winner


def compute_fitnesses(popuation):
    strengths = [
        sum(individual.dominates(other) for other in popuation)
        for individual in popuation
    ]

    for individual in popuation:
        individual.fitness = sum(
            strength_other
            for other, strength_other in zip(popuation, strengths)
            if other.dominates(individual)
        )

    k = int(len(popuation) ** 0.5)  # neighborhood size

    # transform the population to a matrix
    population_matrix = np.array(
        [individual.objectives for individual in popuation]
    )

    # create a KDTree
    tree = KDTree(population_matrix)

    for individual in popuation:
        individual.fitness += 1 / (
            tree.query(individual.objectives, k=k)[0][-1] + 2
        )

    return popuation


def compute_new_archive(population, archive_size):
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

            for i in range(len(new_archive)):
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
    population_size,
    archive_size,
    max_generations,
    verbose=True,
    n_jobs=os.cpu_count(),
):
    # Create the initial population and the archive
    population = [Individual() for _ in range(population_size)]
    archive = []

    # Main loop
    for generation in range(max_generations):
        with Pool(n_jobs) as pool:
            # Compute the fitnesses of the population
            objectives_population = pool.map(evaluation_fun, population)

        for individual, objectives in zip(population, objectives_population):
            individual.objectives = objectives

        # Combine the population and the archive
        combined_population = population + archive

        # Compute the fitnesses of the population
        compute_fitnesses(combined_population)

        # Update the archive
        archive = compute_new_archive(combined_population, archive_size)

        # Select and mutate the next population
        population = [
            tournament_selection(archive, 2).mutate()
            for _ in range(population_size)
        ]

        if verbose:
            print(f"Generation {generation + 1}/{max_generations}")

    return population, archive
