from Node import *
from Priority_queue import *
from dijkstra import *
from queue import Queue
import cv2
import math
import numpy as np
image_URL = "C:\\Users\\ktjos\\Desktop\\Lumpy.jpg"
"""
[(6, 10)]
[(40, 24), (35, 48), (34, 67), (58, 72), (60, 55), (60, 37), (60, 37), (52, 24)]
"""
def create_graph2():
    matrix = [[1,1,4],[2,2,5],[3,3,6]]
    grph = Graph(len(matrix), len(matrix[0]))

    for i in range (len(matrix)):
        for j in range (len(matrix[0])):
            grph.add_node((i,j))

    for i in range (len(matrix)):
        for j in range (len(matrix[0])):
            # try for all four neighbors in order: right, bottom, left, top
            # note here the order in which the neighbors are stored is imp
            if j + 1 < len(matrix[0]):
                grph.add_edge((i,j),(i,j+1),abs(matrix[i][j] - matrix[i][j+1]))
            if i + 1 < len(matrix):
                grph.add_edge((i,j),(i+1,j),abs(matrix[i][j] - matrix[i+1][j]))
            if j -1 >= 0:
                grph.add_edge((i,j),(i,j-1),abs(matrix[i][j] - matrix[i][j-1]))
            if i -1 >=0:
                grph.add_edge((i,j),(i-1,j),abs(matrix[i][j] - matrix[i-1][j]))

    # for keys in grph.vert_list.keys():
    #     print(grph.vert_list[keys])
    #
    # for keys in grph.vert_list.keys():
    #     print(grph.vert_list[keys].list_nbrs)

    for id,node in grph.vert_list.items():
        print("**", id ," " ,node)

    dual = grph.get_dual()

    for id,node in dual.vert_list.items():
        print("**", id ," " ,node)

    print(grph.find_faces(grph.vert_list))


def testing_priority_queue():

    d = {}
    list = [4, 8, 9, 11, 2]
    for i in range (5):
        d[i] = 99

    p = PriorityQueue(d)



    # list= [4,8,2,2,7,5,1,27,11,45,3,22,167,89,0,5,33]
    # list = [4,8,9,11,21,2]

    for num in [0,1,2,3,4]:
        p.insert(num)
    print(p.list)
    print(p.positions)

    for i in range (5):
        d[i] = list[i]
        p.update_priority(i)

    print(p.list)
    print(p.positions)
    for i in range(5):
        print(d[p.pop()])
        print(p.positions)

def create_small_graph():
    g = simple_graph()
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [(1, 2, 2), (1, 4, 3), (2, 3, 9), (2, 5, 2), (3, 6, 1), (4, 5, 4), (4, 7, 5), (5, 6, 14), (5, 8, 6),
             (6, 9, 2),
             (7, 8, 10), (8, 9, 7)]

    nodes = [1,2,3,4]
    edges = [(1, 2, 100.5),(1,3,100.7),(1,4,30.8),(2,4,70.15),(3,4,70.53)]

    for num in nodes:
        g.add_simple_node(num)

    for ed in edges:
        g.add_simple_edge(ed[0],ed[1],ed[2])
        g.add_simple_edge(ed[1], ed[0], ed[2])
    return g

def create_graph():
    """
    Function created for testing dijkstra
    :return:
    """
    g= simple_graph()
        # nodes = [1,2,3,4,5]
        # edges = [(1,2,3),(1,3,10),(2,3,4),(2,4,7),(2,5,3),(3,4,1),(4,5,4)]
        # nodes = [1, 2, 3, 4, 5]
        # edges = [(1, 2, 3), (1, 3, 10), (2, 3, 5), (2, 4, 7), (2, 5, 3), (3, 4, 1),(3, 5, 1), (4, 5, 4)]
    nodes = [1,2,3,4,5,6,7,8,9]
    edges = [(1,2,2),(1,4,3),(2,3,9),(2,5,2),(3,6,1),(4,5,4),(4,7,5),(5,6,14),(5,8,6),(6,9,2),
                (7,8,10),(8,9,7)]

    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    edges = [(1, 2, 5), (2, 3, 7), (3,4,9),
             (1,5,61), (2,6,1), (3,7,4),(4,8,89),
             (5,6,3),(6,7,37),(7,8,44),
             (5,9,2),(6,10,91),(7,11,96),(8,12,54),
             (9,10,1), (10,11,1), (11,12,6),
             (9,13,33), (10,14,6) , (11,15,57), (12,16,103),
             (13,14,17),(14,15,8), (15,16,99)]

    for num in nodes:
        g.add_simple_node(num)

    for ed in edges:
        g.add_simple_edge(ed[0],ed[1],ed[2])
        g.add_simple_edge(ed[1], ed[0], ed[2])
    return g

def test_dijkstra():
    """
    Function created for testing the dijkstra algorithm
    :return:
    """
    graph = create_graph()
    start_node_id = 11
    dist, parent = run_dijkstra(graph, start_node_id)

    for vertice in dist.keys():
        print(vertice, " ", dist[vertice], " ", parent[vertice])


def text_vertex_contraction():
    """
    Function created for testing the dijkstra algorithm
    :return:
    """
    graph = create_graph()
    start_node_id = 11
    dist, parent = run_dijkstra(graph, start_node_id)
    sinks = [10]
    edge_set = set()
    prev = ""

    for sink_vertex in sinks:
        current_vertex = sink_vertex
        prev = ""
        while current_vertex != start_node_id:
            prev = parent[current_vertex]
            edge_set.add((prev,current_vertex))
            current_vertex = prev

    for vertice in dist.keys():
        print(vertice, " ", dist[vertice], " ", parent[vertice])

    print(edge_set)

    graph.contract_edges(edge_set)

    graph.convert_to_list()

    # for v in graph.vert_list:
    #     print(v,":")
    #     print(graph.vert_list[v])
    source = graph.get_simple_node(1)
    sink = graph.get_simple_node('XNode')

    min_cut_edges, source_set_edges = graph.min_cut(source, sink)

    # print(min_cut_edges)
    for nodes in min_cut_edges:
        print('==========================')
        print(nodes[0].get_id(), "edges:", nodes[0])
        print(nodes[1].get_id(), "edges:", nodes[1])
        print('==========================')

    for nodes in source_set_edges:
        print('------------------------------')
        print(nodes.get_id(), "edges:", nodes)
        # print(nodes[1].get_id(), "edges:", nodes[1])
        # print('==========================')

def tp():
    a =[]
    s = str(type(a))
    print(str(s))
    if str(type(a)) == '<class \'list\'>':
        print("success")

def test_queue():
    q = Queue()

    for i in range(5):
        q.put(i)

    while not q.empty():
        print(q.get())
        print(q)
        for i in range(7):
            print(i)
            if i == 5:
                break

def test_edmund_karp():
    g = create_small_graph()
    source = g.get_simple_node(1)
    sink = g.get_simple_node(4)
    g.convert_to_list()
    min_cut_edges = g.min_cut(source, sink)

    print(min_cut_edges)
    for nodes in min_cut_edges:
        print('==========================')
        print(nodes[0].get_id(), "edges:", nodes[0])
        print(nodes[1].get_id(), "edges:", nodes[1])
        print('==========================')

def test_for_100Nodes():
    n = 100

    matrix =[[i for i in range(n)] for _  in range(n)]

    grph = Graph(len(matrix), len(matrix[0]))
    print("creating nodes")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            grph.add_node((i, j))

    print("creating edges")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # try for all four neighbors in order: right, bottom, left, top
            # note here the order in which the neighbors are stored is imp
            if j + 1 < len(matrix[0]):
                grph.add_edge((i, j), (i, j + 1), matrix[i][j])
            if i + 1 < len(matrix):
                grph.add_edge((i, j), (i + 1, j), matrix[i][j])
            if j - 1 >= 0:
                grph.add_edge((i, j), (i, j - 1), matrix[i][j])
            if i - 1 >= 0:
                grph.add_edge((i, j), (i - 1, j), matrix[i][j])

    # for keys in grph.vert_list.keys():
    #     print(grph.vert_list[keys])
    #
    # for keys in grph.vert_list.keys():
    #     print(grph.vert_list[keys].list_nbrs)

    print("creating dual graph")
    dual = grph.get_dual()

    print("testing dijkstra")

    start_node_id = frozenset([0,1,100,101])

    print(dual.vert_list[start_node_id])
    dist, parent = run_dijkstra(dual, start_node_id)

    for vertice in dist.keys():
        print(vertice, " ", dist[vertice], " ", parent[vertice])
    print("Success!!")

def test_image():
    img = cv2.imread(image_URL)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    print(img[0])
    print(len(img[0]))


if __name__ == '__main__':
    # create_graph()
    # tp()
    # testing_priority_queue(
    # tp()
    text_vertex_contraction()
    # test_queue()
    # test_edmund_karp()
    # test_for_100Nodes()
    # create_graph2()
    # test_image()
    # for i in range(4):
    #     print(100000000000000/math.pow(math.e, i*18))