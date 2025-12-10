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
import time
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo

"""
EXAMPLE IN DEVELOPMENT
Network Interdiction Example.
This is an interdiction example for Benders Decomposition.
Max/min interdiction or attacker-defender problems are an important class of bilevel programs classically treated by Benders Decomposition.
Notably, Benders cuts come from the subproblem primal due to a standard formulation dualization trick.
For more on attacker-defender problems and the subproblem dualization see:
Brown, Carlyle, Salmeron, and Wood "Defending Critical Infrastructure"
https://doi.org/10.1287/inte.1060.0252

Specific model and example details adapted from:
Viens, Hart, and Ferris "Extracting Alternative Solutions from Benders Decomposition"
https://arxiv.org/abs/2509.08671
"""


class Network:
    def __init__(self):
        #
        # Data definition section
        #

        # name of the nodes in the directed graph
        nodes = ["s", "t", "a", "b", "c", "d", "e", "f"]

        # name of the source node to match s-t flow format
        source = "s"

        # name of the since node to match s-t flow format
        sink = "t"

        # cost to transit a link/arc
        transit_link_cost = 1

        # added transit cost if link interdicted
        added_transit_cost_if_interdicted = 3

        # flow balance terms
        flow_inputs = {n: 0 for n in nodes}
        flow_inputs[source] = 1
        flow_inputs[sink] = -1

        # arc list, (x,y) format, from x to y
        arcs = [
            ("s", "a"),
            ("s", "b"),
            ("s", "c"),
            ("a", "c"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("d", "f"),
            ("d", "t"),
            ("e", "t"),
            ("f", "t"),
        ]

        # create precessor and successor helper variables
        # directly match FS and RS data
        fs = {u: [v[1] for v in arcs if v[0] == u] for u in nodes}
        rs = {u: [v[0] for v in arcs if v[1] == u] for u in nodes}

        # create link cost and added costs as c[a] and d[a]
        arc_costs = {(a, b): transit_link_cost for (a, b) in arcs}
        added_interdiction_cost = {
            (a, b): added_transit_cost_if_interdicted for (a, b) in arcs
        }


# creates benders master problem model
# takes interdiction costs (r) and interdiction budget (M)
def create_root(network, interdiction_budget=1):
    m = pyo.ConcreteModel()

    # node and arc sets taken from global data
    m.nodes = pyo.Set(initialize=network.nodes)
    m.arcs = pyo.Set(initialize=network.arcs, dimen=2)

    # interdiction choice variables
    # 0 do not interdict
    # 1 do interdict
    m.x = pyo.Var(m.arcs, domain=pyo.Binary)

    # holder variable for cost of defender shortest path
    # need to bound below, default to zero (all costs non-negative)
    m.z = pyo.Var(domain=pyo.Reals)

    # maximize length of shortest defender path
    m.obj = pyo.Objective(expr=m.z, sense=pyo.maximize)

    # interdiction parameters
    m.r = pyo.Param(network.arcs, initialize=network.interdiction_costs)

    # interdiction budget constraint
    m.interdiction_limits = pyo.Constraint(
        expr=sum(m.r[a] * m.x[a] for a in m.arcs) <= interdiction_budget
    )

    # holder object for Benders cuts
    m.cuts = pyo.ConstraintList()

    return m


# method to create the Q value function as a pyomo model
# takes c and d data local
# uses global nodes and arcs data
def create_subproblem(root, network):
    model = pyo.ConcreteModel()

    # node and arc sets
    model.nodes = pyo.Set(initialize=network.nodes)
    model.arcs = pyo.Set(initialize=network.arcs, dimen=2)

    # costs
    model.c = pyo.Param(model.arcs, initialize=network.arc_costs)
    model.d = pyo.Param(model.arcs, initialize=network.added_interdiction_cost)

    # create local copies of the x values that will be updated later
    # paths interdicted 0-1 values, default to 0 (not interdicted)
    model.x = pyo.Var(model.arcs, domain=pyo.Reals)

    # defender path choices
    # continuous between 0 and 1
    # relies on total unimodularity to return integer values
    model.y = pyo.Var(model.arcs, domain=pyo.NonNegativeReals, bounds=(0, 1))

    # create the objective
    # TODO: model.x needs to be a Var in this implementation rather than param
    # That makes this objective bilinear until variables are fixed
    # could apply standard multiplication of binary vars linearization, may make getting easy cuts harder
    # some solvers like gurobi tolerate bilinear terms
    model.obj = pyo.Objective(
        expr=sum(
            (model.c[a] + model.d[a] * model.x[a]) * model.y[a] for a in model.arcs
        ),
        sense=pyo.minimize,
    )

    # create flow balance constraints
    model.flow_inputs = model.Param(model.nodes, initialize=network.flow_inputs)

    def flow_balance_rule(m, node):
        return (
            sum(m.y[(node, v)] for v in network.fs[node])
            - sum(m.y[(u, node)] for u in network.rs[node])
            == model.flow_inputs[node]
        )

    model.flow_balance = pyo.Constraint(model.nodes, rule=flow_balance_rule)

    complicating_vars_map = pyo.ComponentMap()
    for a in network.arcs:
        complicating_vars_map[root.x[a]] = model.x[a]

    return model, complicating_vars_map


def main():
    t0 = time.time()
    network = Network()
    m = create_root(network=network)
    root_vars = list(m.x.values())
    m.benders = BendersCutGenerator()
    m.benders.set_input(root_vars=root_vars, tol=1e-8)
    m.benders.add_subproblem(
        subproblem_fn=create_subproblem,
        subproblem_fn_kwargs={'root': m, 'network': network},
        root_eta=m.eta,
        subproblem_solver='gurobi_persistent',
        already_dualized=True,
    )
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)

    print('{0:<15}{1:<15}{2:<15}'.format('# Cuts', 'Cut' 'Time'))
    for i in range(30):
        res = opt.solve(m, tee=False)
        cuts_added = m.benders.generate_cut()
        for i, c in enumerate(cuts_added):
            opt.add_constraint(c)
        print('{0:<15}{1:<15}{2:<15.2f}'.format(i, str(c.expr), time.time() - t0))
        if len(cuts_added) == 0:
            break


if __name__ == '__main__':
    main()
