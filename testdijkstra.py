from Node import *
import  math
from Priority_queue import *

def create_graph():
    g= simple_graph()
        # nodes = [1,2,3,4,5]
        # edges = [(1,2,3),(1,3,10),(2,3,4),(2,4,7),(2,5,3),(3,4,1),(4,5,4)]
        # nodes = [1, 2, 3, 4, 5]
        # edges = [(1, 2, 3), (1, 3, 10), (2, 3, 5), (2, 4, 7), (2, 5, 3), (3, 4, 1),(3, 5, 1), (4, 5, 4)]
    nodes = [1,2,3,4,5,6,7,8,9]
    edges = [(1,2,2),(1,4,3),(2,3,9),(2,5,2),(3,6,1),(4,5,4),(4,7,5),(5,6,14),(5,8,6),(6,9,2),
                (7,8,10),(8,9,7)]

    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    edges = [(1, 2, 5), (2, 3, 7), (3,4,9),(1,5,61), (2,6,1), (3,7,4),(4,8,89),(5,6,3),(6,7,37),(7,8,44),
             (5,9,2),(5,10,91),(6,11,96),(8,12,54), (9,10,1), (10,11,1), (11,12,6),
             (9,13,33), (10,14,6) , (11,15,57), (12,16,103),
             (13,14,17),(14,15,8), (15,16,99)]

    for num in nodes:
        g.add_simple_node(num)

    for ed in edges:
        g.add_simple_edge(ed[0],ed[1],ed[2])
        g.add_simple_edge(ed[1], ed[0], ed[2])
        # mat_len = 4
        # g = simple_graph()
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[0])):
        #         g.add_simple_node((mat_len*i)+j+1)
        #
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[0])):
        #         # try for all four neighbors in order: right, bottom, left, top
        #         # note here the order in which the neighbors are stored is imp
        #         if j + 1 < len(matrix[0]):
        #             g.add_simple_edge(((mat_len*i)+j+1), ((mat_len*i)+ j + 1+1), matrix[i][j])
        #         if i + 1 < len(matrix):
        #             g.add_simple_edge(((mat_len*i)+j+1), (i + 1, j), matrix[i][j])
        #         if j - 1 >= 0:
        #             grph.add_edge(((mat_len*i)+j+1), (i, j - 1), matrix[i][j])
        #         if i - 1 >= 0:
        #             grph.add_edge(((mat_len*i)+j+1), (i - 1, j), matrix[i][j])


    return g


def test_dijkstra():

    graph = create_graph()

    # for nodes in graph.vert_list:
    #     print(nodes,":",graph.vert_list[nodes])

    run_dijkstra(graph, 11)


def update(source_node, dist, parent, min_heap):
    # the source node will be an object of the class simple_node
    # print(type(source_node))

    for nbrs in source_node.adj_list:
        # print(type(nbrs))
        # print(nbrs)
        # print(dist[nbrs.get_id()] > dist[source_node.get_id()] + source_node.adj_list[nbrs])

        if dist[nbrs.get_id()] > dist[source_node.get_id()] + source_node.adj_list[nbrs]:
            # weight of the neighbor is updated and its priority will be updated in the queue
            dist[nbrs.get_id()] = dist[source_node.get_id()] + source_node.adj_list[nbrs]
            min_heap.update_priority(nbrs.get_id())
            parent[nbrs.get_id()] = source_node.get_id()

    # return dist, parent

def run_dijkstra(graph, source_node):
    dist = {}
    parent = {}
    processed_vertices = set()
    distanes_heap = PriorityQueue(dist)

    for vertices in graph.vert_list:
        dist[vertices] = 999999
        parent[vertices] = None
        distanes_heap.insert(vertices)

    dist[source_node] = 0
    distanes_heap.update_priority(source_node)

    start = graph.get_simple_node(source_node)
    # dist,parent = /
    update(start, dist, parent, distanes_heap)


    for vertices in graph.vert_list:
        if vertices!= source_node:
            min_dist = 999
            min_node = 0
            # print(vertices)
            # for it,val in dist.items():
            #     if min_dist>val and it not in processed_vertices:
            #         min_dist = val
            #         min_node = it
            min_node = distanes_heap.pop()
            while min_node in processed_vertices:
                min_node = distanes_heap.pop()
            print(min_dist, min_node)
            simple_min_node = graph.get_simple_node(min_node)
            # dist,parent = \
            update(simple_min_node, dist, parent, distanes_heap)
            processed_vertices.add(min_node)

    for i in range(len(graph.vert_list)):
        print(str(i+1)," ",dist[i+1]," ",parent[i+1])






def main():
    test_dijkstra()


if __name__ == '__main__':
    main()