import pickle
from math import sqrt
from itertools import combinations
import copy
import numpy as np

class PlaceDB:
    def __init__(self, benchmark):
        route = f'benchmark/{benchmark}/{benchmark}'
        self.benchmark = benchmark
        self.node_info, self.net_info, self.pin_info = {}, {}, {}
        self.port_info = {}
        self.expert_pos = {}
        self.width, self.height = 0, 0
        self.cluster_info = {}
        self.id_to_cluster = []
        self.get_size()
        self.get_node_info(route)
        self.get_net_pin_info(route)
        self.get_expert_pos(route)

    def get_node_info(self, route):
        node_id = 0
        node_range_list = None
        if self.benchmark == "bigblue2" or self.benchmark == "bigblue4":
            node_range_list = pickle.load(open('node_list/node_list_{}_1024.pkl'.format(self.benchmark),'rb'))

        with open(route+'.nodes', 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('\t') or line.startswith(' '):
                    node_line = line.split()
                    if node_line[0].startswith('p'):
                        self.port_info[node_line[0]] = {}
                        self.port_info[node_line[0]]['size_x'] = int(node_line[1])
                        self.port_info[node_line[0]]['size_y'] = int(node_line[2])
                        self.port_info[node_line[0]]['nets'] = []
                    else:
                        self.node_info[node_line[0]] = {}
                        self.node_info[node_line[0]]['x'] = int(node_line[1])
                        self.node_info[node_line[0]]['y'] = int(node_line[2])
                        self.node_info[node_line[0]]['macro'] = False
                        if len(node_line) == 4 and node_range_list is None:
                            self.node_info[node_line[0]]['macro'] = True
                        if node_range_list is not None and node_line[0] in node_range_list:
                            self.node_info[node_line[0]]['macro'] = True
                        self.node_info[node_line[0]]['id'] = node_id
                        self.node_info[node_line[0]]['pins'] = []
                        self.node_info[node_line[0]]['cluster'] = None
                        node_id += 1
                line = f.readline()
    
    def get_net_pin_info(self, route):
        net_id, pin_id = 0, 0
        with open(route+'.nets', 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('NetDegree'):
                    net_line = line.split()
                    self.net_info[net_id] = {}
                    self.net_info[net_id]['net_name'] = net_line[3]
                    self.net_info[net_id]['pins'] = []
                    self.net_info[net_id]['ports'] = []
                    nodes = set()
                    for _ in range(int(net_line[2])):
                        line = f.readline()
                        pin_line = line.split()
                        if pin_line[0].startswith('p'):
                            self.port_info[pin_line[0]]['nets'].append(net_id)
                            self.net_info[net_id]['ports'].append(pin_line[0])
                        else:
                            if pin_line[0] not in nodes:
                                nodes.add(pin_line[0])
                                self.pin_info[pin_id] = {}
                                self.pin_info[pin_id]['node'] = pin_line[0]
                                self.pin_info[pin_id]['net'] =net_id
                                self.pin_info[pin_id]['x'] = float(pin_line[-2])
                                self.pin_info[pin_id]['y'] = float(pin_line[-1])
                                self.net_info[net_id]['pins'].append(pin_id)
                                self.node_info[pin_line[0]]['pins'].append(pin_id)
                        pin_id += 1
                    net_id += 1

                line = f.readline()
    
    def get_size(self):
        if (self.benchmark) == 'adaptec1':
            self.width = 11589
            self.height = 11589
        if (self.benchmark) == 'adaptec2':
            self.width = 15244
            self.height = 15244
        if (self.benchmark) == 'adaptec3':
            self.width = 23386
            self.height = 23386
        if (self.benchmark) == 'adaptec4':
            self.width = 23386
            self.height = 23386
        if (self.benchmark) == 'bigblue1':
            self.width = 11589
            self.height = 11589
        if (self.benchmark) == 'bigblue2':
            self.width = 23084
            self.height = 23084
        if (self.benchmark) == 'bigblue3':
            self.width = 27868
            self.height = 27868
        if (self.benchmark) == 'bigblue4':
            self.width = 32386
            self.height = 32386
    
    def get_expert_pos(self, route):
        with open(route+'.pl','r') as f:
            line = f.readline()
            while line:
                pos_line = line.split()
                if len(pos_line) == 5:
                    if pos_line[0].startswith('p'):
                        self.port_info[pos_line[0]]['x'] = float(pos_line[1])
                        self.port_info[pos_line[0]]['y'] = float(pos_line[2])
                    else:
                        self.expert_pos[pos_line[0]] = {}
                        self.expert_pos[pos_line[0]]['x'] = float(pos_line[1])
                        self.expert_pos[pos_line[0]]['y'] = float(pos_line[2])
                line = f.readline()

class ClusterDB:
    def __init__(self, placedb: PlaceDB):
        self.benchmark = placedb.benchmark
        self.width = placedb.width
        self.height = placedb.height
        self.cluster_info = {}
        self.net_info = {}
        self.pin_info = {}
        self.expert_pos = placedb.expert_pos
        self.port_info = placedb.port_info
        self.id_to_cluster = []
        self.adjacency = {}

        self.get_cluster_info(placedb)
        self.get_net_pin_info(placedb)

        self.cluster_net_num = {}
        for c_id in self.cluster_info:
            self.cluster_net_num[c_id] = 0
        for net_id in self.net_info:
            if len(self.net_info[net_id]['clusters']) > 1:
                for c_id in self.net_info[net_id]['clusters']:
                    self.cluster_net_num[c_id] += 1
        print(len(self.net_info))

        self.importance_sort()
    
    def get_cluster_info(self, placedb:PlaceDB):
        # route = f'/cluster/{placedb.benchmark}-dict.pkl'
        # with open(route, 'rb') as file:
        #     clusters = pickle.load(file)
        clusters = {}

        for c_id in clusters:
            self.cluster_info[c_id] = {}
            self.cluster_info[c_id]['id'] = None
            self.cluster_info[c_id]['nodes'] = clusters[c_id]
            pin = placedb.node_info[clusters[c_id][0]]['pins'][0]
            self.cluster_info[c_id]['pins'] = [pin]
            c_size = 0
            for node_name in clusters[c_id]:
                c_size += placedb.node_info[node_name]['x'] * placedb.node_info[node_name]['y']
                placedb.node_info[node_name]['cluster'] = c_id
            self.cluster_info[c_id]['x'] = sqrt(c_size)
            self.cluster_info[c_id]['y'] = sqrt(c_size)
        
        c_id = len(self.cluster_info)
        for node in placedb.node_info:
            if placedb.node_info[node]['macro']:
                self.cluster_info[c_id] = {}
                self.cluster_info[c_id]['id'] = None
                self.cluster_info[c_id]['nodes'] = [node]
                self.cluster_info[c_id]['pins'] = placedb.node_info[node]['pins']
                self.cluster_info[c_id]['x'] = placedb.node_info[node]['x']
                self.cluster_info[c_id]['y'] = placedb.node_info[node]['y']
                placedb.node_info[node]['cluster'] = c_id
                c_id += 1
    
    def get_net_pin_info(self, placedb:PlaceDB):
        net_id1 = 0
        for net_id2 in placedb.net_info:
            self.net_info[net_id1] = {}
            self.net_info[net_id1]['pins'] = set()
            self.net_info[net_id1]['ports'] = placedb.net_info[net_id2]['ports'].copy()
            self.net_info[net_id1]['clusters'] = set()
            self.net_info[net_id1]['net_name'] = placedb.net_info[net_id2]['net_name']
            for pin1 in placedb.net_info[net_id2]['pins']:
                node = placedb.pin_info[pin1]['node']
                cluster = placedb.node_info[node]['cluster']
                if cluster is not None:
                    self.net_info[net_id1]['clusters'].add(cluster)
                if placedb.node_info[node]['macro']:
                    pin_id = pin1
                elif cluster is not None:
                    pin_id = self.cluster_info[cluster]['pins'][0]
                else:
                    pin_id = None

                if pin_id is not None:
                    self.net_info[net_id1]['pins'].add(pin_id)

                    if pin_id not in self.pin_info:
                        self.pin_info[pin_id] = {}
                        self.pin_info[pin_id]['cluster'] = cluster
                        self.pin_info[pin_id]['nets'] = set()
                        if placedb.node_info[node]['macro']:
                            self.pin_info[pin_id]['x'] = placedb.pin_info[pin_id]['x']
                            self.pin_info[pin_id]['y'] = placedb.pin_info[pin_id]['y']
                        else:
                            self.pin_info[pin_id]['x'] = 0
                            self.pin_info[pin_id]['y'] = 0
                    self.pin_info[pin_id]['nets'].add(net_id1) 
            
            if len(self.net_info[net_id1]['pins']) + len(self.net_info[net_id1]['ports']) <= 1:
                for pin_id in self.net_info[net_id1]['pins']:
                    self.pin_info[pin_id]['nets'].remove(net_id1) 
                    if len(self.pin_info[pin_id]['nets']) == 0:
                        del self.pin_info[pin_id]
                del self.net_info[net_id1]
            else:
                net_id1 += 1 
        print(net_id1)
    
    def importance_sort(self):
        for net_id in self.net_info:
            if len(self.net_info[net_id]['clusters']) < 2:
                continue
            for c_id1, c_id2 in list(combinations(self.net_info[net_id]['clusters'], 2)):
                if c_id1 not in self.adjacency:
                    self.adjacency[c_id1] = set()
                if c_id2 not in self.adjacency:
                    self.adjacency[c_id2] = set()
                self.adjacency[c_id1].add(c_id2)
                self.adjacency[c_id2].add(c_id1)
        
        cluster_net_num = copy.deepcopy(self.cluster_net_num)

        visited_cluster = set()
        add_cluster = max(cluster_net_num, key = lambda v: cluster_net_num[v]*1000 + self.cluster_info[v]['x']*self.cluster_info[v]['y'])
        visited_cluster.add(add_cluster)
        self.id_to_cluster.append(add_cluster)
        cluster_net_num.pop(add_cluster)

        while len(self.id_to_cluster) < len(self.cluster_info):
            candidates = {}
            for cluster_name in visited_cluster:
                if cluster_name not in self.adjacency:
                    continue
                for cluster_name_2 in self.adjacency[cluster_name]:
                    if cluster_name_2 in visited_cluster:
                        continue
                    if cluster_name_2 not in candidates:
                        candidates[cluster_name_2] = 0
                    candidates[cluster_name_2] += 1
            for cluster_name in self.cluster_info:
                if cluster_name not in candidates and cluster_name not in visited_cluster:
                    candidates[cluster_name] = 0
            if len(candidates) > 0:
                candidates = {key: candidates[key] for key in sorted(candidates.keys())}
                if self.benchmark != 'ariane':
                    if self.benchmark == "bigblue3":
                        add_cluster = max(candidates, key = lambda v: candidates[v]*1 + cluster_net_num[v]*100000 +\
                            self.cluster_info[v]['x']*self.cluster_info[v]['y'] * 1)
                    else:
                        add_cluster = max(candidates, key = lambda v: candidates[v]*1 + cluster_net_num[v]*1000 +\
                            self.cluster_info[v]['x']*self.cluster_info[v]['y'] * 1)
                else:
                    add_cluster = max(candidates, key = lambda v: candidates[v]*1 + cluster_net_num[v]*1 +\
                        self.cluster_info[v]['x']*self.cluster_info[v]['y']*1)

            else:
                if self.benchmark != 'ariane':
                    if self.benchmark == "bigblue3":
                        add_cluster = max(cluster_net_num, key = lambda v: cluster_net_num[v]*100000 + self.cluster_info[v]['x']*self.cluster_info[v]['y']*1)
                    else:
                        add_cluster = max(cluster_net_num, key = lambda v: cluster_net_num[v]*1000 + self.cluster_info[v]['x']*self.cluster_info[v]['y']*1)
                else:
                    add_cluster = max(cluster_net_num, key = lambda v: cluster_net_num[v]*1 + self.cluster_info[v]['x']*self.cluster_info[v]['y']*1)

            visited_cluster.add(add_cluster)
            self.id_to_cluster.append(add_cluster) 
            cluster_net_num.pop(add_cluster)

        for i,c_id in enumerate(self.id_to_cluster):
            self.cluster_info[c_id]['id'] = i