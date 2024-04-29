from main import compare_convergences
import numpy as np

mu,sigma = (50,10)
X = np.random.rand(10000)
Y = np.random.normal(mu,sigma,10000)
data = np.vstack((X,Y))
compare_convergences([(0,0,1),(0,10,2),(0,20,4),(0,30,6),(0,40,8),(0,50,10)], data, 10000, False, True, 'results/simulation_study/')