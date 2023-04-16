import numpy as np
import ioh
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count

import sys
import argparse
import warnings
import os

from modcma import ModularCMAES

from bayes_optim import BO, RealSpace
from bayes_optim.surrogate import GaussianProcess, RandomForest


def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(50, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results

class Algorithm_Evaluator():
    def __init__(self, alg, model, aq, opt):
        self.alg = alg
        self.model_choice = model
        self.aq = aq
        self.opt = opt
        self.budget = 100 #times dim


    def __call__(self, func, seed):
        np.random.seed(int(seed))
        if self.alg == "BO":
            dim = func.meta_data.n_variables
            space = RealSpace([-5, 5]) * dim  # create the search space

            if self.model_choice == "GP":
                model = GaussianProcess(space)
            elif self.model_choice == "RF":
                model = RandomForest(levels=space.levels)
            
            opt = BO(
                search_space=space,
                obj_fun=func,
                model=model,
                DoE_size=10,                         # number of initial sample points
                max_FEs=self.budget,                         # maximal function evaluation
                acquisition_fun=self.aq,
                acquisition_optimization={"optimizer": self.opt, 'max_FEs': 1000*dim},
                verbose=False
            )
            opt.run()
        else: 
            print(f"{self.alg} Does not exist! ________")
        
def run_optimizer(temp):
    
    algname, model, aq, opt, fid, dim, type_ = temp

    # print(algname, fid)
    
    algorithm = Algorithm_Evaluator(algname, model, aq, opt)


    logger = ioh.logger.Analyzer(root="data/", folder_name=f"{algname}-{model}-{aq}_F{fid}_{dim}D_{type_.name}", algorithm_name=f"{algname}-{model}-{aq}_{type_.name}", )

    for iid in list(range(1,6)) + list(range(101,111)):
        func = ioh.get_problem(fid, dimension=dim, instance=iid, problem_class=type_) #in ioh < 0.3.9, problem_class -> problem_type
        # func.enforce_bounds(np.inf)
        func.attach_logger(logger)
        try:
            algorithm(func, iid)
        except:
            print(f"Failed run on {fid} {iid}")
        print(f"Done with run {model}-{aq}-{opt} {fid} {iid}\n")
        func.reset()
    logger.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    fids = range(1,25)
    
    algnames = ['BO']
    models = ["GP", "RF"]
    aqs = ["EI", "MGFI", "UCB", "EpsilonPI"]
    opts =  ["OnePlusOne_Cholesky_CMA"]
    dims = [5,20]
    tpyes = [ioh.ProblemClass.SBOX, ioh.ProblemClass.BBOB]#in ioh < 0.3.9, problemClass -> problemType
    
    args = product(algnames, models, aqs, opts, fids, dims, tpyes)

    runParallelFunction(run_optimizer, args)
