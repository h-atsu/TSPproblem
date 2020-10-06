import csv
import networkx as nx
import pulp
import sys
import matplotlib.pyplot as plt
import math
from itertools import product

size = 20

p = [(0, 0)]
n = 1

# input
with open("dataset/tsp_dataset_{}.csv".format(size)) as f:
    reader = csv.reader(f)
    for (cnt, h) in enumerate(reader, start=1):
        if cnt >= 3:
            break
    for row in reader:
        row = [float(n) for n in row]
        p.append(tuple(row))
        n += 1


vlist = [i for i in range(n)]
G = nx.Graph()
G.add_nodes_from(vlist)

nodes = list(G.nodes())
edges = [(nodes[i], nodes[j])
         for (i, j) in product(range(n), range(n)) if nodes[i] != nodes[j]]

D = {}
for i in range(n):
    for j in range(n):
        D[i, j] = math.sqrt((p[i][0]-p[j][0])**2 + (p[i][1]-p[j][1])**2)

prob = pulp.LpProblem('TSP', pulp.LpMinimize)

x = {(u, v): pulp.LpVariable('x_{0}_{1}'.format(u, v), lowBound=0, cat='Binary')
     for (u, v) in product(range(n), range(n)) if u != v}
u = {i: pulp.LpVariable('u_{0}'.format(i), lowBound=0,
                        cat='Integer') for i in range(n)}
prob += u[0] == 1
for i in range(1, n):
    prob += u[i] <= n
    prob += u[i] >= 2

prob += pulp.lpSum(D[i, j] * x[i, j]
                   for (i, j) in product(range(n), range(n)) if i != j)
for i in range(n):
    prob += pulp.lpSum(x[i, j] for j in range(n) if i != j) == 1
for i in range(n):
    prob += pulp.lpSum(x[j, i] for j in range(n) if i != j) == 1
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            prob += u[i] - u[j] + (n+1) * x[i, j] <= n


prob.solve()


tours = []
for i in range(n):
    for j in range(n):
        if i != j:
            if pulp.value(x[i, j]) == 1:
                tours.append([i, j])
                print(i, j)
G.add_edges_from(tours)

nx.draw_networkx(G, pos=p, node_color='k', node_size=10, with_labels=False)
plt.axis('off')
plt.show()
