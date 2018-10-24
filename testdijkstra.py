from Node import *
import  math

def create_graph():
    g= simple_graph()
    # nodes = [1,2,3,4,5]
    # edges = [(1,2,3),(1,3,10),(2,3,4),(2,4,7),(2,5,3),(3,4,1),(4,5,4)]
    # nodes = [1, 2, 3, 4, 5]
    # edges = [(1, 2, 3), (1, 3, 10), (2, 3, 5), (2, 4, 7), (2, 5, 3), (3, 4, 1),(3, 5, 1), (4, 5, 4)]
    nodes = [1,2,3,4,5,6,7,8,9]
    edges = [(1,2,2),(1,4,3),(2,3,9),(2,5,2),(3,6,1),(4,5,4),(4,7,5),(5,6,14),(5,8,6),(6,9,2),
                (7,8,10),(8,9,7)]
    for num in nodes:
        g.add_simple_node(num)

    for ed in edges:
        g.add_simple_edge(ed[0],ed[1],ed[2])
        g.add_simple_edge(ed[1], ed[0], ed[2])

    return g


def test_dijkstra():

    graph = create_graph()

    # for nodes in graph.vert_list:
    #     print(nodes,":",graph.vert_list[nodes])

    run_dijkstra(graph, 1)


def update(source_node, dist, parent):
    # the source node will be an object of the class simple_node
    # print(type(source_node))

    for nbrs in source_node.adj_list:
        # print(type(nbrs))
        # print(nbrs)
        # print(dist[nbrs.get_id()] > dist[source_node.get_id()] + source_node.adj_list[nbrs])

        if dist[nbrs.get_id()] > dist[source_node.get_id()] + source_node.adj_list[nbrs]:
            dist[nbrs.get_id()] = dist[source_node.get_id()] + source_node.adj_list[nbrs]
            parent[nbrs.get_id()] = source_node.get_id()

    # return dist, parent

def run_dijkstra(graph, source_node):
    dist = {}
    parent = {}
    processed_vertices = set()

    for vertices in graph.vert_list:
        dist[vertices] = math.inf
        parent[vertices] = None
    dist[source_node] = 0
    start = graph.get_simple_node(source_node)
    # dist,parent = /
    update(start, dist, parent)

    for vertices in graph.vert_list:
        if vertices!= source_node:
            min_dist = 999
            min_node = 0
            print(vertices)
            for it,val in dist.items():
                if min_dist>val and it not in processed_vertices:
                    min_dist = val
                    min_node = it
            print(min_dist, min_node)
            simple_min_node = graph.get_simple_node(min_node)
            # dist,parent = \
            update(simple_min_node, dist, parent)
            processed_vertices.add(min_node)

    print(dist)
    print(parent)





def main():
    test_dijkstra()


if __name__ == '__main__':
    main()