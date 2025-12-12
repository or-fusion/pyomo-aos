#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo
import time

#
# EXAMPLE IN DEVELOPMENT
# Newsvendor example adapted from
#
# A Tutorial on Stochastic Programming
# Alexander Shapiro∗ and Andy Philpott†
# March 21, 2007
# https://www.epoc.org.nz/papers/ShapiroTutorialSP.pdf
#

import pyomo.environ as pyo
from forestlib.sp import stochastic_program


#
# Data for a simple newsvendor example
#
class Newsvendor:
    def __init__(self):
        c = 1.0
        b = 1.5
        h = 0.1
        scenario_demand = {1: 15, 2: 60, 3: 72, 4: 78, 5: 82}
        scenarios = scenario_demand.keys()
        scenario_probabilities = {i: 1 / len(scenarios) for i in scenarios}


# creates benders master problem model for newsvendor
def create_root(newsvendor):
    M = pyo.ConcreteModel()

    M.x = pyo.Var(within=pyo.NonNegativeReals)
    M.scenarios = pyo.Set(initialize=newsvendor.scenarios, ordered=True)

    M.eta = pyo.Var(M.scenarios)
    M.obj = pyo.Objective(expr=sum(M.eta.values()))
    return M


def create_subproblem(root, newsvendor, scenario):
    M = pyo.ConcreteModel()

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    b = newsvendor.b
    c = newsvendor.c
    h = newsvendor.h
    d = newsvendor.scenario_demand[scenario]

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.obj = pyo.Objective(expr=newsvendor.scenario_probabilities[scenario] * M.y)

    complicating_vars_map = pyo.ComponentMap()
    complicating_vars_map[root.x] = M.x

    return M, complicating_vars_map


def main():

    t0 = time.time()
    newsvendor = Newsvendor()
    m = create_root(newsvendor=newsvendor)
    root_vars = list(m.x.values())
    m.benders = BendersCutGenerator()
    m.benders.set_input(root_vars=root_vars, tol=1e-8)
    for s in newsvendor.scenarios:
        subproblem_fn_kwargs = dict()
        subproblem_fn_kwargs['root'] = m
        subproblem_fn_kwargs['newsvendor'] = newsvendor
        subproblem_fn_kwargs['scenario'] = s
        m.benders.add_subproblem(
            subproblem_fn=create_subproblem,
            subproblem_fn_kwargs=subproblem_fn_kwargs,
            root_eta=m.eta[s],
            subproblem_solver='gurobi_persistent',
        )
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)

    print('{0:<15}{1:<15}{2:<15}'.format('# Cuts', 'x', 'Time'))
    for i in range(30):
        res = opt.solve(tee=False, save_results=False)
        cuts_added = m.benders.generate_cut()
        for c in cuts_added:
            opt.add_constraint(c)
        print(
            '{0:<15}{1:<15.2f}{2:<15.2f}'.format(
                len(cuts_added), pyo.value(m.x), time.time() - t0
            )
        )
        if len(cuts_added) == 0:
            break


if __name__ == '__main__':
    main()
