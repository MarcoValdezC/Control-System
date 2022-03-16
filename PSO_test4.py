import numpy as np
import matplotlib.pyplot as plt


class PSO(object):
    def __init__(self, population_size, max_steps):
        self.w = 0.6  # peso de inercia
        self.c1 = self.c2 = 2
        self.population_size = population_size  # número de enjambre de partículas
        self.dim = 2  # Dimensión del espacio de búsqueda
        self.max_steps = max_steps  # número de iteraciones
        self.x_bound = [-10, 10]  # rango de espacio de solución
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim))  # Inicializar la posición del enjambre de partículas
        # Inicializar la velocidad del enjambre de partículas
        self.v = np.random.rand(self.population_size, self.dim)
        fitness = self.calculate_fitness(self.x)
        
        #print(fitness)
        self.p = self.x  # La mejor posición del individuo
        #print(self.p)
        self.pg = self.x[np.argmin(fitness)]  # mejor posición global
        #print(self.pg)
        self.individual_best_fitness = fitness  # La aptitud óptima del individuo
        self.global_best_fitness = np.max(
            fitness)  # Mejor estado físico global

    def calculate_fitness(self, x):
        return np.sum(np.square(x), axis=1)

    def evolve(self):
        fig = plt.figure()
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # Actualizar velocidad y peso
            self.v = self.w*self.v+self.c1*r1 * \
                (self.p-self.x)+self.c2*r2*(self.pg-self.x)
            self.x = self.v + self.x
            plt.clf()
            plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')
            plt.xlim(self.x_bound[0], self.x_bound[1])
            plt.ylim(self.x_bound[0], self.x_bound[1])
            plt.pause(0.01)
            fitness = self.calculate_fitness(self.x)
            # Individuos que necesitan ser actualizados
            update_id = np.greater(self.individual_best_fitness, fitness)
            print(update_id)
            self.p[update_id] = self.x[update_id]
            print(len(self.p[update_id]))
            self.individual_best_fitness[update_id] = fitness[update_id]
            # La nueva generación tiene un estado físico más pequeño, así que actualice el estado físico y la posición óptimos globales
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            print('best fitness: %.5f, mean fitness: %.5f' %
                  (self.global_best_fitness, np.mean(fitness)))


pso = PSO(100, 100)
pso.evolve()
plt.show()
