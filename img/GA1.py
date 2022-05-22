import numpy

# Parameter initialization
genes = 2
chromosomes = 10
mattingPoolSize = 6
offspringSize = chromosomes - mattingPoolSize
lb = -5
ub = 5
populationSize = (chromosomes, genes)
generations = 3

# Population initialization
population = numpy.random.uniform(lb, ub, populationSize)

for generation in range(generations):
    print(("Generation:", generation+1))
    fitness = numpy.sum(population*population, axis=1)
    print("\npopulation")
    print(population)
    print("\nfitness calcuation")
    print(fitness)
    # Following statement will create an empty two dimensional array to store parents
    parents = numpy.empty((mattingPoolSize, population.shape[1]))

    # A loop to extract one parent in each iteration
    for p in range(mattingPoolSize):
        # Finding index of fittest chromosome in the population
        fittestIndex = numpy.where(fitness == numpy.max(fitness))
        # Extracting index of fittest chromosome
        fittestIndex = fittestIndex[0][0]
        print(fittestIndex)
        # Copying fittest chromosome into parents array
        parents[p, :] = population[fittestIndex, :]
        # Changing fitness of fittest chromosome to avoid reselection of that chromosome
        fitness[fittestIndex] = -1
        print(fitness)
    print("\nParents:")
    print(parents)

#     # Following statement will create an empty two dimensional array to store offspring
#     offspring = numpy.empty((offspringSize, population.shape[1]))
#     for k in range(offspringSize):
#         # Determining the crossover point
#         crossoverPoint = numpy.random.randint(0, genes)

#         # Index of the first parent.
#         parent1Index = k % parents.shape[0]

#         # Index of the second.
#         parent2Index = (k+1) % parents.shape[0]

#         # Extracting first half of the offspring
#         offspring[k, 0: crossoverPoint] = parents[parent1Index, 0: crossoverPoint]

#         # Extracting second half of the offspring
#         offspring[k, crossoverPoint:] = parents[parent2Index, crossoverPoint:]
#     print("\nOffspring after crossover:")
#     print(offspring)

#     # Implementation of random initialization mutation.
#     for index in range(offspring.shape[0]):
#         randomIndex = numpy.random.randint(1, genes)
#         randomValue = numpy.random.uniform(lb, ub, 1)
#         offspring[index, randomIndex] = offspring[index,
#                                                   randomIndex] + randomValue
#     print("\n Offspring after Mutation")
#     print(offspring)

#     population[0:parents.shape[0], :] = parents
#     population[parents.shape[0]:, :] = offspring
#     print("\nNew Population for next generation:")
#     print(population)

# fitness = numpy.sum(population*population, axis=1)
# fittestIndex = numpy.where(fitness == numpy.max(fitness))
# # Extracting index of fittest chromosome
# fittestIndex = fittestIndex[0][0]
# # Getting Best chromosome
# fittestInd = population[fittestIndex, :]
# bestFitness = fitness[fittestIndex]
# print("\nBest Individual:")
# print(fittestInd)
# print("\nBest Individual's Fitness:")
# print(bestFitness)
