import random
import math
import matplotlib.pyplot as plt

from spea2 import spea2


# the Individual class that represents a candidate solution
# the only requirement for the class is that it has the method:
#   - .mutate() -> Individual
#       returns a new individual that is a mutation of the current individual
class Individual:
    def __init__(self, x=None):
        if x is None:
            self.x = random.uniform(-10, 10)
        else:
            self.x = x

    def mutate(self):
        return Individual(self.x + random.gauss(0, 1))


# the evaluation function that takes an individual and returns a tuple of objectives
def evaluation_function(individual):
    objective1 = individual.x**2
    objective2 = (individual.x - 2) ** 2

    return objective1, objective2


# the dominates function takes two tuples of objectives as returned by the evaluation function
# and returns True if the first tuple dominates the second tuple
# usually, domination means that:
# the first tuple is not worse than the second tuple in all objectives and is better in at least one objective

# depending on the problem, the definition of domination can be different
# moreover, one can switch the sign of the objectives to maximize them instead of minimizing, as in the example


# most of the time, the implementation of this function will look like this:
def dominates(objectives1, objectives2):
    return all(o1 <= o2 for o1, o2 in zip(objectives1, objectives2)) and any(
        o1 < o2 for o1, o2 in zip(objectives1, objectives2)
    )


# the main function that runs the algorithm
population, archive = spea2(
    Individual,
    evaluation_function,
    dominates,
    population_size=100,
    archive_size=10,
    max_generations=100,
    verbose=False,
    n_jobs=2,
    callback=None,
)

# the archive contains the best solutions found during the run, while the population contains the last generation
# the archive is usually the most interesting part of the output

# the archive is a list of IndividualWrapper objects, which contain the individual and its objectives
# to get the individual, use the .individual attribute
# to get the objectives, use the .objectives attribute

# plot the archive
objectives1 = [individual.objectives[0] for individual in archive]
objectives2 = [individual.objectives[1] for individual in archive]

plt.scatter(objectives1, objectives2)
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.title("Pareto front")
plt.show()
