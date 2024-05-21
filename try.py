import PyGP
from PyGP import Population

from PyGP.utils import IterRun, time_record
import time
import random
import numpy as np
from PyGP import DCurve, Curve
import math
from PyGP import Dataset
import pandas as pd
import matplotlib.pyplot as plt

# from pmlb import regression_dataset_names, fetch_data
#
# print(regression_dataset_names)
#
# x, y = fetch_data('adult', return_X_y=True)
# print(x, y)


from PyGP import dataset_dict as d, dataset_save

dataset_name = "Nonic"
pmlb_dataset = Dataset(*d[dataset_name])
pmlb_dataset.set_dataset('./Dataset/pmlb/datasets')


def get_pmlb_data(data_id, seed_id):
    from PyGP import dataset_dict as d
    return pmlb_dataset.from_pmlb(data_id, seed_id, True, False, False)


def get_data(isPMLB=False):
    from PyGP import dataset_dict as d

    dataset = Dataset(*d[dataset_name])

    if isPMLB:
        return dataset.from_pmlb("adult")
    else:
        return dataset.get_data()

def train(run_id, data_id=None):
    if data_id is not None:
        res = get_pmlb_data(data_id, run_id)
        if res is None:
            return None
        (curve_size, n_terms, range_, data_size, data, fitness, range_curve, data_curve, fitness_curve,
            data_candidate,
            fitness_candidate, data_candidate_1, fitness_candidate_1) = res
    else:
        (curve_size, n_terms, range_, data_size, data, fitness, range_curve, data_curve, fitness_curve, data_candidate,
         fitness_candidate, data_candidate_1, fitness_candidate_1) = get_data()
    f_s = np.sort(fitness)
    curve_draw = Curve(range_curve, [f_s[0] - math.fabs(0.1 * f_s[-1]), f_s[-1] + math.fabs(0.1 * f_s[-1])])

    print('data_rg: ', range_, random.randint(0, 100))
    d_curve_draw = None
    
    pop = Population(pop_size=PyGP.POP_SIZE, cross_rate=0.9, mut_rate=0.9,
                     function_set=['mul', 'add', 'sub', 'div', 'sin', 'cos', 'log', 'exp'], seed=1234)
    # pop.func_register(func_name='ops', func=reg_test, arity=4)
    data_f_num = 0
    for i in range(n_terms):
        data_f_num += int((range_[i][1] - range_[i][0]) / 100)
    if data_f_num < data_size and data_f_num > 0 and n_terms > 10000000000:
        data_f_s = True
        data_f_num = int(data_size / 10) if int(data_size / 10) > data_f_num else data_f_num
        (data_f, fitness_f) = PyGP.data_filter(data, fitness, range_curve, data_f_num)
        print('------------------------------------------------', data_f, fitness_f)
        pop.initDataset(data_f, fitness_f, range_curve)
        d_curve_draw.add_points([data_f, fitness_f])
    else:
        data_f_s = False
        pop.initDataset(data, fitness, range_curve)
        # d_curve_draw.add_points([data, fitness])

    start = time.time()
    pop.initialization(initial_method='half-and-half', init_depth=[2, 8])
    PyGP.LIBRARY_SUPPLEMENT_INTERVAL = 1
    PyGP.LIBRARY_SUPPLEMENT_NUM = 1
    pop.bpselect_depth_ = 1

    
    func_set = [['mul', 'add', 'sub', 'div'],
            ['mul', 'add', 'sub', 'div', 'log'],
            ['mul', 'add', 'sub', 'div', 'sin', 'cos'],
            ['mul', 'add', 'sub', 'div', 'sin', 'cos', 'log'],
            ['mul', 'add', 'sub', 'div', 'sin', 'cos', 'log', 'exp']]

    for i in range(10):
        # if np.random.randint(4) > 2:
        #     new_fset = func_set[0]
        # else:
        #     new_fset = func_set[1]
        # pop.pop_dict[pop.pop_id]['funcs'] = funcs
        

        pop.progs = []
        for j in range(pop.pop_size):
            
            new_fset = func_set[np.random.randint(len(func_set))]
            funcs_name = new_fset
            funcs = PyGP.FunctionSet(type(PyGP.TreeNode))
            funcs.init(funcs_name)

            if isinstance(pop.init_depth, int):
                depth = pop.init_depth  # random.randint(2, init_depth + 1)
            else:
                depth = random.randint(2, 4)
            pop.progs.append(PyGP.Program(pop.pop_id, j, init_depth=depth, funcs=funcs))
        
        if PyGP.SEMANTIC_SIGN:
            pmask = range(pop.pop_size)
            pop.backpSelect(PyGP.LIBRARY_SUPPLEMENT_NUM, pmask)
        pop.execution()
        # print('=======================semantics size: ', pop.semantics.get_lib_size(), '==========================')
    print("len(pop.semantics.library_data)", pop.semantics.library_size())
    pop.bpselect_depth_ = None
    pop.progs = []
    # pop.method = 'half-and-half'


    new_fset = func_set[0]
    funcs_name = new_fset
    funcs = PyGP.FunctionSet(type(PyGP.TreeNode))
    funcs.init(funcs_name)

    for j in range(pop.pop_size):
        if isinstance(pop.init_depth, int):
            depth = pop.init_depth  # random.randint(2, init_depth + 1)
        else:
            depth = random.randint(2, 4)
        pop.progs.append(PyGP.Program(pop.pop_id, j, init_depth=depth, funcs=funcs))

    PyGP.LIBRARY_SUPPLEMENT_INTERVAL = 10
    PyGP.LIBRARY_SUPPLEMENT_NUM = 20
    pop.genetic_register('crossover', PyGP.SMT_Weight_Crossover_LV2, pop.pop_size)
    pop.register('crossover__', PyGP.SMT_Weight_Crossover_LV2, pop.pop_size)
    pop.register('const_optimize', PyGP.ConstOptimization, pop.pop_size)
    
    pop.slt_time = 0

    if PyGP.SEMANTIC_SIGN:
        
        pmask = np.argsort(pop.child_fitness)
        pmask = pmask[:50]
        pop.backpSelect(PyGP.LIBRARY_SUPPLEMENT_NUM, pmask)
    pop.execution(0)

    # print("random-1:  ", random.randint(0, 100), np.random.randint(0, 100))
    end = time.time()
    iteration = 201
    (res, R2, _, _) = pop.verify(data_curve, fitness_curve, inverse_transform=False)

    # output = pop.getOutput(0, curve_size)
    # d_curve_draw.append_data(data_curve, output, 0, res[0])

    pop.selection()

    print('run time：', end - start, 'second', 'aver_size: ', pop.getAverSize())

    depth_limit = pop.depth_limit
    oper_limit = 5 * 10 ** 8
    keep_time = 0
    train_res = 0
    eld_fit = pop.child_fitness[0]
    for i in range(iteration):
        start = time.time()
        # print("random0:  ", random.randint(0, 100), np.random.randint(0, 100))
        mutate = False
        if oper_limit <= 0 or pop.child_fitness[0] < 1e-12:
            break
        print("=================================================", "run_id: ", run_id, " iter: " + str(i),
              "remain_operand: ", oper_limit, pop.getAverSize(), len(pop.fitness), pop.depth_limit,
              "=================================================")
        nan_arg = None
        nan_arg_cur = None
        time_s = []
        r_mut_rate = 1 # - float(0.2 * i) / float(iteration)
        if np.random.uniform(0, 1) < r_mut_rate:  # random.uniform(0, 1) < r_mut_rate:
            time_record(time_s, pop.crossover, pop.funcs, pop.depth_limit)
        else:
            mut_rate = 1  # pop.mut_rate + float((1. - pop.mut_rate) * i) / float(iteration * 2)
            time_record(time_s, pop.mutation, mut_rate, pop.funcs)
            mutate = True
        
        if PyGP.SEMANTIC_SIGN:
            
            pmask = np.argsort(pop.child_fitness)
            pmask = pmask[:int(pop.pop_size / 10)]
            pop.backpSelect(PyGP.LIBRARY_SUPPLEMENT_NUM, pmask)
        time_record(time_s, pop.execution)

        time_record(time_s, pop.selection, nan_arg, nan_arg_cur)

        print('depth, fitness, exp: ', pop.progs[0].depth, '(', pop.child_fitness[0], pop.child_prlt[0], ')',
              pop.progs[0].print_exp())

        if pop.child_fitness[0] < eld_fit:  # i % 10 == 0:
            eld_fit = pop.child_fitness[0]
            (res, R2, _, r_cpu) = pop.verify(data_candidate, fitness_candidate, inverse_transform=False)

            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!R2: ', R2[0],
                  res[0], r_cpu[0], '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        end = time.time()
        keep_time += end - start
        print("=================================================",
              'total_time:{}, crossover:{}, mutation:{}, execution:{}, selection:{}'.format(keep_time,
                                                                                                    0 if mutate else round(
                                                                                                        time_s[
                                                                                                            0],
                                                                                                        4),
                                                                                                    round(time_s[
                                                                                                              0],
                                                                                                          4) if mutate else 0,
                                                                                                    round(time_s[
                                                                                                              1],
                                                                                                          4),
                                                                                                    round(time_s[
                                                                                                              2],
                                                                                                          4)),
              '秒', 'aver_size: ', pop.getAverSize(), 'best prog size: ', pop.progs[0].size,
              "=================================================")


        oper_limit -= pop.getAverSize() * pop.pop_size
        train_res = pop.child_fitness[0]

    (train_res, train_R2, _, train_res_cpu) = pop.verify(data, fitness, inverse_transform=False)

    (res, R2, _, r_cpu) = pop.verify(data_candidate, fitness_candidate, inverse_transform=False)

    f_s = np.sort(fitness_candidate)

    (res_1, R2_1, _, _) = pop.verify(data_candidate_1, fitness_candidate_1, inverse_transform=False)

    pop.pool.close()
    pop.pool.join()
    print('res_1: ', res_1[0])
    print('R2_1: ', R2_1[0])


    return (
        (res[0], res_1[0]), (train_res[0], train_R2[0], train_res_cpu[0]), (R2[0], R2_1[0]), pop.progs[0].print_exp(),
        d_curve_draw, keep_time, pop.progs[0].size)


import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-run_id', dest='run_id', type=int,
                        help='the index of run')
    parser.add_argument('-job_id', dest='job_id', type=int, default=None,
                        help='the index of dataset')
    parser.add_argument('-file_name', dest='save_file', type=str, default="PMLB_res.txt",
                        help='the index of dataset')
    args = parser.parse_args()
    next = False
    test_min = 100000000
    
    if args.save_file == './result/gausson_test_0.1.txt':
        PyGP.NOISE = 0.1
        PyGP.SEMANTIC_CDD = 20
        PyGP.INTERVAL_COMPUTE = False
    if args.save_file == './result/PMLB_res_fri_test_2.txt':
        PyGP.INTERVAL_COMPUTE = False
        PyGP.DEPTH_MAX_SIZE = 15
        PyGP.MAX_TREE_SIZE = 500
    if args.save_file == './result/PMLB_res_fri_test_nocrg.txt':
        PyGP.CRO_STG = 1
    if args.save_file == './result/PMLB_res_fri_test_noslt.txt':
        PyGP.SEMANTIC_CDD = 1
    print('NOISE: ', PyGP.NOISE)
    res = train(args.run_id, args.job_id)
    if res is None:
        next = True

        with open(args.save_file, 'a+') as f:
            f.write("{0}\n".format("Sth wrong....."))
        exit(-1)
    else:
        (test, train_res, R2, exp, d_curve_draw, keep_time, prog_size) = res
    if (test[1] < test_min):
        # d_curve_draw.dynamic_curve(pmlb_dataset.datasets[args.job_id].split('\\')[-1].split('.tsv.gz')[0])
        test_min = test[1]
    plt.clf()
    with open(args.save_file, 'a+') as f:
        f.write(
            "{0} {1} {2} {3} {4} {5} {6}\n".format(R2[0], test[0], keep_time, train_res[0], train_res[1], train_res[2],
                                                   prog_size))
        f.write("{0}\n".format(exp))