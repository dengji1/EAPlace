import numpy as np
from canvas import Canvas
from placedb import ClusterDB
from place_modules import greedy_place, place
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import multiprocessing
import os

def get_conj_nets(net, net_bounds, indices):
    for j in range(indices.shape[0]):
        if net_bounds[net]['x_min'] < indices[j, 0] < net_bounds[net]['x_max']:
            if net_bounds[net]['y_min'] < indices[j, 1] < net_bounds[net]['y_max']:
                return net
    return None

class Solution:
    def __init__(self, chip:Canvas, w_r=0, w_o=0):
        self.position = chip.node_pos
        self.overlap = chip.overlap
        self.rudy = chip.rudy
        self.net_bounds = chip.net_bounds
        self.ratio = chip.ratio
        self.hpwl = chip.hpwl
        self.w_r = w_r
        self.w_o = w_o
    
    def compute_module_score(self, placedb:ClusterDB):
        hpwl = np.zeros((len(placedb.id_to_cluster),), dtype=np.float32)
        rudy = np.zeros((len(placedb.id_to_cluster),), dtype=np.float32)
        overlap = np.zeros((len(placedb.id_to_cluster),), dtype=np.float32)
        value = np.percentile(self.rudy, 98)
        indices = np.argwhere(self.rudy >= value)

        st = time.time()
        net_set = set()
        with multiprocessing.Pool() as pool:
            results = pool.starmap(get_conj_nets, [(net, self.net_bounds, indices) for net in self.net_bounds])
        for result in results:
            if result is not None:
                net_set.add(result)

        st = time.time()
        for i, c_id in enumerate(placedb.id_to_cluster):
            x, y = self.position[c_id]['x'], self.position[c_id]['y']
            size_x, size_y = self.position[c_id]['size_x'], self.position[c_id]['size_y']
            overlap[i] = np.sum(self.overlap[x:x+size_x][y:y+size_y])
            overlap[i] /= (size_x * size_y)

            net_cnt1, net_cnt2 = 0, 0
            for pin_id in placedb.cluster_info[c_id]['pins']:
                if pin_id not in placedb.pin_info:
                    continue
                for net_id in placedb.pin_info[pin_id]['nets']:
                    net_x = (self.net_bounds[net_id]['x_min'] + self.net_bounds[net_id]['x_max']) / 2
                    net_y = (self.net_bounds[net_id]['y_min'] + self.net_bounds[net_id]['y_max']) / 2
                    pin_x = round((placedb.pin_info[pin_id]['x'] + size_x / 2 + x * self.ratio) / self.ratio)
                    pin_y = round((placedb.pin_info[pin_id]['y'] + size_y / 2 + y * self.ratio) / self.ratio)
                    hpwl[i] += abs(pin_x - net_x) + abs(pin_y - net_y)
                    net_cnt1 += 1

                    if net_id in net_set:
                        net_cnt2 += 1
                        rudy[i] += 1
                
                if net_cnt1 > 0:
                    hpwl[i] /= net_cnt1
                if net_cnt2 > 0:
                    rudy[i] /= net_cnt2
            
        if hpwl.max() > 0:
            hpwl /= hpwl.max()
        if rudy.max() > 0:
            rudy /= rudy.max()
        if overlap.max() > 0:
            overlap /= overlap.max()
        score = hpwl + self.w_r * rudy + self.w_o * overlap
        return score
    
    def select_module(self, placedb:ClusterDB):
        score = self.compute_module_score(placedb)
        score = np.exp(score) / np.sum(np.exp(score))
        indices = np.random.choice(len(score), size=int(len(score)*0.6), replace=False, p=score)
        c_idxs = []
        indices.sort()
        for idx in indices:
            c_idxs.append(placedb.id_to_cluster[idx])
        return c_idxs
    
    def adjust_position(self, placedb:ClusterDB, c_idxs):
        chip = Canvas(placedb)
        fixed_ids = list(set(placedb.id_to_cluster) - set(c_idxs))
        chip = place(chip, fixed_ids, self.position)
        chip = greedy_place(chip, c_idxs)
        return Solution(chip, self.w_r, self.w_o)
    
    def save_placement(self, placedb:ClusterDB):
        os.makedirs(f'pl', exist_ok=True)
        file_path = f"pl/{placedb.benchmark}.pl"
        with open(file_path, 'w') as fwrite:
            for c_id in self.position:
                if len(placedb.cluster_info[c_id]['nodes']) == 1:
                    node_name = placedb.cluster_info[c_id]['nodes'][0]
                    x, y = self.position[c_id]['x'], self.position[c_id]['y']
                    x = x * self.ratio
                    y = y * self.ratio
                    fwrite.write('{}\t{}\t{}\t:\tN\n'.format(node_name, x, y))
    
    def plot_placement(self, replace_idxs=[]):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        for node_name in list(self.position.keys()):
            x, y = self.position[node_name]['x'], self.position[node_name]['y']
            size_x, size_y = self.position[node_name]['size_x'], self.position[node_name]['size_y']
            color = 'b'
            if node_name in replace_idxs:
                color = 'g'
            ax1.add_patch(
                patches.Rectangle(
                    (x/224, y/224), 
                    size_x/224,  
                    size_y/224, linewidth=1, edgecolor='k',facecolor = color
                )
            )
        fig1.savefig('plot.png', dpi=90, bbox_inches='tight')
        plt.close()

class Population:
    def __init__(self, max_size, w_r=0, w_o=0):
        self.max_size = max_size
        self.solutions = []
        self.metrics = {
            'hpwl':[],
            'rudy': [],
            'overlap':[]
        }
        self.w_r = w_r
        self.w_o = w_o

    def add(self, solution:Solution):
        self.solutions.append(solution)
        self.metrics['hpwl'].append(solution.hpwl)
        self.metrics['rudy'].append(solution.rudy.max())
        self.metrics['overlap'].append(solution.overlap.sum())
        self.exclude()
    
    def select(self):
        score = self.calculate_score() / 1e3
        score = np.exp(score) / np.sum(np.exp(score))
        idx = np.random.choice(len(score), p=score)
        return self.solutions[idx]
    
    def exclude(self):
        while len(self.solutions) > self.max_size:
            score = self.calculate_score()
            idx = np.argmin(score)
            del self.solutions[idx]
            del self.metrics['hpwl'][idx]
            del self.metrics['rudy'][idx]
            del self.metrics['overlap'][idx]
    
    def get_best_solution(self):
        score = self.calculate_score()
        idx = np.argmax(score)
        return self.solutions[idx]
    
    def calculate_score(self):
        hpwl = np.array(self.metrics['hpwl']) / 1e5
        rudy = np.array(self.metrics['rudy']) / 1e2
        overlap = np.array(self.metrics['overlap']) / 1e2
        score = -(hpwl + rudy * self.w_r + overlap * self.w_o)
        return score



