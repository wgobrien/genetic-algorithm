from ga import GeneticAlgorithm
import test_functions as testf

if __name__=='__main__':
    ga = GeneticAlgorithm(testf.rosenbrock, ['x','y','z'], [(-5,5),(-5,5),(-5,5)], pop_size=100)
    b, n = ga.run(mode='minimize'
                  , select='rank'
                  , boltzmann=True
                  , generations=1000
                  , exploration=.35
                  , keep_top=5
                  , verbose=True)
    ga.export(b)
