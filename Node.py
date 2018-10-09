

class Graph:

    __slots__ = ['vert_list', 'num_vertices']

    def __init__(self):
        self.vert_list = {}
        self.num_vertices = 0

    def add_node(self,key):
        """
        :param key:  here the key will be the i and j coordinate of the nodes
        :return:
        """
        pass

    def add_edge(self, node1, node2, cost):
        pass


class node:

    __slots__ = ['i', 'j' ,'adj_list']

    def __init__(self,i,j):
        self.i = i
        self.j = j
        self.adj_list = {}

    def add_edge(self,nbr,weight = 1):
        pass

    def get_coordinates(self):
        return self.i, self.j

    def get_next_nbr(self):
        pass

    def get_nbrs(self):
        return self.adj_list.keys()
