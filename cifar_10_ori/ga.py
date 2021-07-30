import numpy as np
from geneticalgorithm import geneticalgorithm as soga
from test_resnet_backdoor_chain import test_func
import traceback
import atexit

def exit_handler():
    traceback.print_stack()

atexit.register(exit_handler)


def f(X):
    return -test_func(X)


varbound = np.array([[0, 15]] + [[0, 15]]  + [[0, 31]] + [[0, 63]])

model = soga(function=f, dimension=4, variable_type='int', variable_boundaries=varbound)

model.run()


