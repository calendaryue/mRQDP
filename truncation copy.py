import sys
import numpy as np
import networkx as nx
from operator import itemgetter
import random
from collections import defaultdict
#该代码包括两部分：1.导入文件，存储节点和边到图里
#2.输入图，进行阶段操作，输出图类型
def project_degree_based(g, theta, noreplace=True, order='random', edge_order=None):
  g2 = g.copy()
  edge_set = list(g2.edges())
  node_set = g2.nodes()
  degree_dict = {}
  h = nx.MultiGraph()
  h.add_nodes_from(node_set)

  if edge_order is not None:
    edge_set = edge_order
  else:
    if order == 'random':
      random.shuffle(edge_set)
    elif order == 'low' or order == 'high':
      dkey = build_compare_key(edge_set, g.degree())
      sort_dkey = sorted(dkey.items(), key=itemgetter(1))
      edge_set, value = zip(*sort_dkey)
      if order == 'high':
        edge_set = edge_set[::-1]
#如果添加边后，两节点均超过阈值则舍弃，一节点超过则
  for v1,v2,data in g2.edges(data=True):
    if check_degree(degree_dict, v1,theta) and \
       check_degree(degree_dict, v2,theta):
      # m1 = add_dict(degree_dict, v1, edge_weight, theta)
      # m2 = add_dict(degree_dict, v2, edge_weight, theta)
      edge_weight = data['weight'] # Get the weight of the edge in g
      # if (m1==0) and (m2==0):
      h.add_edge(v1, v2, weight=edge_weight)
      update_dict(degree_dict, v1, edge_weight)
      update_dict(degree_dict, v2, edge_weight)

  return h

def check_degree(deg_dict, v, t):
  if v in deg_dict and deg_dict[v] >= t:
    return False
  else:
    return True

def update_dict(deg_dict, v, edge_weight):
  if v not in deg_dict:
    deg_dict[v] = edge_weight
  else:
    deg_dict[v] += edge_weight
  return deg_dict
  
# def update_dict(deg_dict, v, edge_weight):
#     deg_dict[v] += edge_weight

# def add_dict(deg_dict, v, edge_weight, theta):
#   if v not in deg_dict:
#     deg_dict[v] = edge_weight
#     modify = 0
#   elif (deg_dict[v] + edge_weight)<=theta:
#     deg_dict[v] += theta
#     modify = 0
#   else:
#     modify = deg_dict[v] + edge_weight - theta
    
  return modify


def remove_edges(g, h):
  g.remove_edges_from(h.edges())
  deg = g.degree()
  to_remove = [n for n in deg if deg[n] == 0]
  # to_remove = [n for n in deg if n in deg and deg[n] == 0]
  g.remove_nodes_from(to_remove)
  return g

def build_compare_key(edge_set, degree_dict):
  dkey = {}
  for e in edge_set:
    u, v = e
    const = (u+v)/100000
    du = degree_dict[u]
    dv = degree_dict[v]
    dkey[e] = max(du*100000+dv+const, dv*100000+du+const)
  return dkey

def print_info(g):
  print('Graph info:')
  print('# of nodes:', g.number_of_nodes())
  print('# of edges:', g.number_of_edges())
  print('# of CC:', nx.number_connected_components(g))
  print('max degree:', max(dict(g.degree()).values()))
  print('avg degree:', g.number_of_edges()*2/g.number_of_nodes())

def process_input(filename, verbose=False):
  g = read_graph(filename)
  # graphs = list(nx.connected_component_subgraphs(g))
  graphs = [g.subgraph(c).copy() for c in nx.connected_components(g)]
  dataname = filename[5:-4]
  if verbose:
    print_info(g)
    print(f'Data name = "{dataname}"')
  edge_weights = nx.get_edge_attributes(g, 'weight')
  total_weight = sum(edge_weights.values())
  print(total_weight)
  # return graphs[0], dataname
  return g, dataname

def read_graph(filename):
  """ Data format is 'v1 v2' on each line. """
  f = open(filename, 'r')
  data = f.read()
  lines = data.split('\n')

  g = nx.MultiGraph(gid=filename)
  node_dict = {}
  vid = 0

  for line in lines:
    if line != "":
      line = line.strip()
      (contri,v1, v2) = line.split()
      if v1 not in node_dict:
        node_dict[v1] = vid
        vid += 1
      if v2 not in node_dict:
        node_dict[v2] = vid
        vid += 1
      g.add_edge(node_dict[v1], node_dict[v2], weight = float(contri))

  f.close()
  return g
#../Information/TPCH/sc_0/Q3_0.txt
def main():
  print ('Process input...')
  g, dataname = process_input(sys.argv[1], verbose=True)

  edge_weights = nx.get_edge_attributes(g, 'weight')
  total_weight = sum(edge_weights.values())
  print(total_weight)
  h = project_degree_based(g,20)
  print_info(h)
  print(total_weight)

if __name__ == '__main__':
    main()
