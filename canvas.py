from placedb import ClusterDB
from math import ceil
import numpy as np
from numba import jit

class Canvas:
    def __init__(self, placedb:ClusterDB, grid=224, fixed_nodes=False):
        self.grid = grid
        self.ratio = placedb.width / grid
        self.node_pos = {} # 以grid为单位
        self.net_bounds = {} # 以grid为单位
        self.placedb = placedb
        self.overlap = np.zeros((grid, grid), dtype=np.float32)
        self.rudy = np.zeros((grid, grid), dtype=np.float32)
        self.hpwl = 0
        for port in self.placedb.port_info:
            for net in self.placedb.port_info[port]['nets']:
                port_x = round((self.placedb.port_info[port]['x'] + self.placedb.port_info[port]['size_x'] / 2) / self.ratio)
                port_y = round((self.placedb.port_info[port]['y'] + self.placedb.port_info[port]['size_y'] / 2) / self.ratio)
                if net not in self.net_bounds:
                    self.net_bounds[net] = {}
                    self.net_bounds[net]['x_min'] = port_x
                    self.net_bounds[net]['x_max'] = port_x
                    self.net_bounds[net]['y_min'] = port_y
                    self.net_bounds[net]['y_max'] = port_y
                else:
                    self.net_bounds[net]['x_min'] = min(self.net_bounds[net]['x_min'], port_x, 0)
                    self.net_bounds[net]['x_max'] = max(self.net_bounds[net]['x_max'], port_x, self.grid)
                    self.net_bounds[net]['y_min'] = min(self.net_bounds[net]['y_min'], port_x, 0)
                    self.net_bounds[net]['y_max'] = max(self.net_bounds[net]['y_max'], port_x, self.grid)
        if fixed_nodes:
            for cluster in placedb.fixed_clusters:
                assert len(placedb.cluster_info[cluster]['nodes']) == 1
                node = placedb.cluster_info[cluster]['nodes'][0]
                x, y = placedb.expert_pos[node]['x'], placedb.expert_pos[node]['y']
                x = ceil(x / self.ratio)
                y = ceil(y / self.ratio)
                self.place_module(cluster, x, y)

    def place_module(self, cluster_id, x, y):
        for pin in self.placedb.cluster_info[cluster_id]['pins']:
            if pin not in self.placedb.pin_info:
                continue
            for net in self.placedb.pin_info[pin]['nets']:
                pin_x = round((self.placedb.pin_info[pin]['x'] + self.placedb.cluster_info[cluster_id]['x'] / 2 + x * self.ratio) / self.ratio)
                pin_y = round((self.placedb.pin_info[pin]['y'] + self.placedb.cluster_info[cluster_id]['y'] / 2 + y * self.ratio) / self.ratio)

                if net not in self.net_bounds:
                    self.net_bounds[net] = {}
                    self.net_bounds[net]['x_min'] = pin_x
                    self.net_bounds[net]['x_max'] = pin_x
                    self.net_bounds[net]['y_min'] = pin_y
                    self.net_bounds[net]['y_max'] = pin_y
                else:
                    x_min, x_max, y_min, y_max = self.net_bounds[net]['x_min'], self.net_bounds[net]['x_max'], self.net_bounds[net]['y_min'], self.net_bounds[net]['y_max']
                    self.rudy[x_min:x_max+1][y_min:y_max+1] -= 1 / (x_max + 1 - x_min) + 1 / (y_max + 1 - y_min)
                    if pin_x < self.net_bounds[net]['x_min']:
                        self.hpwl += self.net_bounds[net]['x_min'] - pin_x
                        self.net_bounds[net]['x_min'] = pin_x
                    if pin_x > self.net_bounds[net]['x_max']:
                        self.hpwl += pin_x - self.net_bounds[net]['x_max']
                        self.net_bounds[net]['x_max'] = pin_x
                    if pin_y < self.net_bounds[net]['y_min']:
                        self.hpwl += self.net_bounds[net]['y_min'] - pin_y
                        self.net_bounds[net]['y_min'] = pin_y
                    if pin_y > self.net_bounds[net]['y_max']:
                        self.hpwl += pin_y - self.net_bounds[net]['y_max']
                        self.net_bounds[net]['y_max'] = pin_y
                        
                x_min, x_max, y_min, y_max = self.net_bounds[net]['x_min'], self.net_bounds[net]['x_max'], self.net_bounds[net]['y_min'], self.net_bounds[net]['y_max']
                self.rudy[x_min:x_max+1][y_min:y_max+1] += 1 / (x_max + 1 - x_min) + 1 / (y_max + 1 - y_min)
        
        size_x = ceil(self.placedb.cluster_info[cluster_id]['x'] / self.ratio)
        size_y = ceil(self.placedb.cluster_info[cluster_id]['y'] / self.ratio)
        self.overlap[x:x+size_x, y:y+size_y] += 1

        self.node_pos[cluster_id] = {'x': x, 'y': y, 'size_x': size_x, 'size_y': size_y}
    
    def get_rudy(self):
        for net in self.net_bounds:
            x_min = self.net_bounds[net]['x_min']
            x_max = self.net_bounds[net]['x_max']
            y_min = self.net_bounds[net]['y_min']
            y_max = self.net_bounds[net]['y_max']
            self.rudy[x_min:x_max+1][y_min:y_max+1] += 1 / (x_max + 1 - x_min) + 1 / (y_max + 1 - y_min)
    
    def get_hpwl_increment(self, cluster_id):
        hpwl_inc = np.zeros((self.grid, self.grid), dtype=np.float32)
        for pin in self.placedb.cluster_info[cluster_id]['pins']:
            if pin not in self.placedb.pin_info:
                continue
            for net in self.placedb.pin_info[pin]['nets']:
                if net in self.net_bounds:
                    offset_x = round((self.placedb.pin_info[pin]['x'] + self.placedb.cluster_info[cluster_id]['x'] / 2) / self.ratio)
                    offset_y = round((self.placedb.pin_info[pin]['y'] + self.placedb.cluster_info[cluster_id]['y'] / 2) / self.ratio)
                    x_min = self.net_bounds[net]['x_min']
                    x_max = self.net_bounds[net]['x_max']
                    y_min = self.net_bounds[net]['y_min']
                    y_max = self.net_bounds[net]['y_max']
                    if x_min - offset_x > 0:
                        vec = np.arange(x_min-offset_x, 0, -1, dtype=np.float32)
                        if x_min - offset_x  > self.grid:
                            vec = vec[:self.grid]
                        hpwl_inc[0 : min(x_min-offset_x, self.grid) :] += np.tile(vec[:, np.newaxis],(1, self.grid))
                    if x_max - offset_x < self.grid:
                        vec = np.arange(0, self.grid + offset_x - x_max, dtype=np.float32)
                        if x_max - offset_x < 0:
                            vec = vec[-x_max + offset_x:]
                        hpwl_inc[max(x_max-offset_x, 0) : self.grid, :] += np.tile(vec[:, np.newaxis],(1, self.grid))
                    if y_min - offset_y > 0:
                        vec = np.arange(y_min-offset_y, 0, -1, dtype=np.float32)
                        if y_min - offset_y > self.grid:
                            vec = vec[:self.grid]
                        hpwl_inc[:, 0 : min(y_min-offset_y, self.grid)] += np.tile(vec[np.newaxis, :], (self.grid, 1))
                    if y_max - offset_y < self.grid:
                        vec = np.arange(0, self.grid + offset_y - y_max, dtype=np.float32)
                        if y_max - offset_y < 0:
                            vec = vec[-y_max + offset_y:]
                        hpwl_inc[:, max(y_max-offset_y, 0) : self.grid] += np.tile(vec[np.newaxis, :], (self.grid, 1))
        return hpwl_inc
    
    def get_pos(self, cluster_id):
        pos = np.zeros((self.grid, self.grid), dtype=np.float32)
        w = ceil(self.placedb.cluster_info[cluster_id]['x'] / self.ratio)
        h = ceil(self.placedb.cluster_info[cluster_id]['y'] / self.ratio)

        pos[self.grid - w + 1:self.grid, :] = 1
        pos[:, self.grid - h + 1:self.grid] = 1

        for cluster_id1 in self.node_pos:
            cluster_x = self.node_pos[cluster_id1]['size_x']
            cluster_y = self.node_pos[cluster_id1]['size_y']
            node_pos_x = self.node_pos[cluster_id1]['x']
            node_pos_y = self.node_pos[cluster_id1]['y']

            x_min = max(0, node_pos_x - w + 1)
            x_max = min(node_pos_x + cluster_x, self.grid)
            y_min = max(0, node_pos_y - h + 1)
            y_max = min(node_pos_y + cluster_y, self.grid)
            pos[x_min : x_max, y_min : y_max] = 1
        return pos
    
    def get_overlap(self, cluster_id):
        overlap = np.zeros((self.grid, self.grid), dtype=np.float32)
        w = ceil(self.placedb.cluster_info[cluster_id]['x'] / self.ratio)
        h = ceil(self.placedb.cluster_info[cluster_id]['y'] / self.ratio)
        grid = np.arange(self.grid, dtype=np.float32)
        for cluster_id1 in self.node_pos:
            x_left = self.node_pos[cluster_id1]['x']
            y_left = self.node_pos[cluster_id1]['y']
            x_right = self.node_pos[cluster_id1]['x']+ceil(self.placedb.cluster_info[cluster_id1]['x'] / self.ratio)
            y_right = self.node_pos[cluster_id1]['y']+ceil(self.placedb.cluster_info[cluster_id1]['y'] / self.ratio)

            left_max = np.maximum(x_left, grid)
            right_min = np.minimum(x_right, grid + w)
            overlap_x = np.maximum(right_min - left_max, 0)

            left_max = np.maximum(y_left, grid)
            right_min = np.minimum(y_right, grid + h)
            overlap_y = np.maximum(right_min - left_max, 0)

            overlap += overlap_x[:, np.newaxis] * overlap_y[np.newaxis, :]

        overlap[self.grid-w+1:self.grid, :] = np.inf
        overlap[:, self.grid-h+1:self.grid] = np.inf
        return overlap
