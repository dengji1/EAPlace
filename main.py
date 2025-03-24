import argparse
from placedb import PlaceDB, ClusterDB
from canvas import Canvas
import numpy as np
import time
from datetime import datetime
from population import Population, Solution
from place_modules import greedy_place
import random
import os

parser = argparse.ArgumentParser(description='hyperparameters')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--init_iter', type=int, default=20)
parser.add_argument('--max_iter', type=int, default=3000)
parser.add_argument('--grid', type=int, default=224)
parser.add_argument('--enable_log', type=bool, default=False)
parser.add_argument('--benchmark', type=str, default='adaptec1')
parser.add_argument('--N', type=int, help='population size', default = 5)
parser.add_argument('--rudy_weight', type=float, default=0)
parser.add_argument('--overlap_weight', type=float, default=0)
args = parser.parse_args()

def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class Log:
    def __init__(self, benchmark, enable_log=True):
        self.enable_log = enable_log
        self.file_name = f'logs/{benchmark}/{args.seed}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
        if self.enable_log:
            os.makedirs(f'logs/{benchmark}', exist_ok=True)
            self.fopen = open(self.file_name, 'w')
            self.fopen.write('time,score,hpwl,rudy,overlap\n')
    
    def write_line(self, pop:Population):
        if self.enable_log:
            string = f'{time.time()},'
            score = list(pop.calculate_score())
            string += " ".join(map(str, score))
            string += ','
            string += " ".join(map(str, pop.metrics['hpwl']))
            string += ','
            string += " ".join(map(str, pop.metrics['rudy']))
            string += ','
            string += " ".join(map(str, pop.metrics['overlap']))
            string += '\n'
            self.fopen.write(string)
            self.fopen.flush()

    def close_log(self):
        if self.enable_log:
            self.fopen.close()

if __name__ == '__main__':
    set_seed(args.seed)

    placedb = PlaceDB(benchmark=args.benchmark)
    clusterdb = ClusterDB(placedb)
    del placedb
    log = Log(args.benchmark, args.enable_log)
    population = Population(args.N, args.rudy_weight, args.overlap_weight)

    t = time.time()
    for i in range(args.init_iter):
        canvas = Canvas(placedb=clusterdb, grid=args.grid)
        greedy_place(canvas, clusterdb.id_to_cluster)
        solution = Solution(canvas)
        hpwl1, overlap1, rudy1 = solution.hpwl, solution.overlap.sum(), solution.rudy.max()
        population.add(solution)
        log.write_line(population)
        best_solution = population.get_best_solution()
        print(f'--- init iter {i}/{args.init_iter} ---')
        print(f'solution hpwl {hpwl1}')
        print(f'best hpwl {best_solution.hpwl}')

    for i in range(args.max_iter):
        solution = population.select()
        replace_idxs = solution.select_module(clusterdb)
        new_solution = solution.adjust_position(clusterdb, replace_idxs)
        hpwl1, overlap1, rudy1 = new_solution.hpwl, new_solution.overlap.sum(), new_solution.rudy.max()
        population.add(new_solution)
        log.write_line(population)
        best_solution = population.get_best_solution()
        print(f'--- iter {i}/{args.max_iter} ---')
        print(f'solution hpwl {hpwl1}')
        print(f'best hpwl {best_solution.hpwl}')

    best_solution = population.get_best_solution()
    best_solution.save_placement(clusterdb)
    log.close_log()
    print(f'total time {time.time()-t} s')
