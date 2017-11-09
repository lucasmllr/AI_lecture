import numpy as np
from queue import PriorityQueue
import networkx as nx
from copy import deepcopy

#funciton to createnetwrokx graph of some cities in romania and their distances
def makeRomaniaGraph():

    nodes = [(1, {'name':'Oradea'}), (2, {'name':'Zerind'}), (3, {'name':'Arad'}),\
    (4, {'name':'Timisoara'}), (5, {'name':'Lugoj'}), (6, {'name':'Mehadia'}),\
    (7, {'name':'Drobeta'}), (8, {'name':'Sibiu'}), (9, {'name':'Rimnicu Vilcea'}),\
    (10, {'name':'Craiova'}), (11, {'name':'Fagaras'}), (12, {'name':'Pitesti'}),\
    (13, {'name':'Giurgio'}), (14, {'name':'Bucharest'}), (15, {'name':'Urziceni'}),\
    (16, {'name':'Neamt'}), (17, {'name':'Iasi'}), (18, {'name':'Vaslui'}),\
    (19, {'name':'Hirsova'}), (20, {'name':'Eforie'})]

    edges = [(1, 2, {'weight':71}), (2, 3, {'weight':75}), (3, 4, {'weight':118}),
    (4, 5, {'weight':111}), (5, 6, {'weight':70}), (6, 7, {'weight':75}),
    (1, 8, {'weight':151}), (3, 8, {'weight':140}), (7, 10, {'weight':120}),
    (8, 11, {'weight':99}), (8, 9, {'weight':80}), (9, 12, {'weight':97}),
    (9, 10, {'weight':146}), (11, 14, {'weight':211}), (12, 14, {'weight':101}),
    (10, 12, {'weight':138}), (13, 14, {'weight':90}), (14, 15, {'weight':85}),
    (15, 18, {'weight':142}), (17, 18, {'weight':92}), (16, 17, {'weight':87}),
    (15, 19, {'weight':98}), (19, 20, {'weight':86})]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G

#function finding closest path on a networkx graph with edge weights and nodes that
#have a name attribute
def Dijkstra(graph, start, destination):
    #initialize priority queue
    Q = PriorityQueue()
    #initialize additional attributes in nodes
    for i in graph.nodes():
        graph.node[i]['visited'] = False
        graph.node[i]['path'] = []
        if graph.node[i]['name'] == start:
            graph.node[i]['distance'] = 0
            Q.put((0, i))
        else: #all other nodes are initialized with a non existing distance
            graph.node[i]['distance'] = None


    while not Q.empty():
        #setting closest node as current
        current = Q.get()[1]
        #return path to destination once it is reached. This is always the shortest distance.
        if graph.node[current]['name'] == destination:
            #adding destination to path
            graph.node[current]['path'].append(current)
            return graph.node[current]['distance'], [graph.node[i]['name'] for i in graph.node[current]['path']],

        else:
            #looping through all neighbours of current node
            for neighbor in graph.neighbors(current):
                #alrady visited nodes are excluded, because they were already reached on the shortest paths
                if not graph.node[neighbor]['visited']:
                    weight = graph.edge[current][neighbor]['weight']
                    newDistance = graph.node[current]['distance'] + weight
                    #checking wheather stored distance is shorter than current one
                    #this happens when the neighbour node has already been reached on a different path
                    #but has not been visited itself, i.e. has not been a current node
                    if graph.node[neighbor]['distance'] == None or \
                    newDistance < graph.node[neighbor]['distance']:
                        #updating distance
                        graph.node[neighbor]['distance'] = newDistance
                        #updating path
                        currentPath = deepcopy(graph.node[current]['path'])
                        currentPath.append(current)
                        graph.node[neighbor]['path'] = currentPath
                        #placing neighbor with updated distance into queue
                        Q.put((graph.node[neighbor]['distance'], neighbor))
            #marking current node as visited
            graph.node[current]['visited'] = True
    #if queue is empty before destination is reached there is no connection between start and destination
    print("no connection from start to destination.")
    return

if __name__ == '__main__':
    G = makeRomaniaGraph()
    print(Dijkstra(G, 'Arad', 'Bucharest'))
