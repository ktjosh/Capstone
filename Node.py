class Graph:
    __slots__ = ['num_columns', 'vert_list', 'num_vertices', 'num_rows']

    def __init__(self, num_rows, num_columns):
        self.vert_list = {}
        self.num_vertices = 0
        self.num_columns = num_columns
        self.num_rows = num_rows

    def add_node(self, coordinates):
        """
        :param key:  here the key will be the i and j coordinate of the nodes
        :return:
        """
        if coordinates not in self.vert_list:
            new_node = node(coordinates, self.num_columns)
            key = new_node.get_id()
            self.vert_list[key] = new_node
            self.num_vertices += 1

    def add_edge(self, node1_coordinates, node2_coordinates, cost=1):
        """
        the definition might change based on the later requirements
        :param node1:
        :param node2:
        :param cost:
        :return:
        """
        node1_key = self.get_node_id(node1_coordinates)
        node2_key = self.get_node_id(node2_coordinates)
        node1 = self.vert_list[node1_key]
        node2 = self.vert_list[node2_key]
        node1.add_nbr(node2, cost)

    def get_node(self, key):
        return self.vert_list[key]

    def get_dual(self):
        """
        function finds dual graph of a given graph
        :param vertlist:
        :return:
        """

        dual_vertices = self.find_faces(self.vert_list)
        dual = simple_graph()
        for vertices in dual_vertices:
            dual.add_simple_node(vertices)

        for vert1 in dual_vertices:
            for vert2 in dual_vertices:
                if vert1 != vert2:
                    common = vert1 & vert2
                    if (len(common) > 1):
                        # not decided what weight will be used but a weight will be used
                        dual.add_simple_edge(vert1, vert2)
                        dual.add_simple_edge(vert2, vert1)

        return dual

    def find_faces(self, vertlist):
        faces_set = set()

        for nodes in vertlist.keys():
            (i, j) = vertlist[nodes].get_coordinates()

            if i + 1 != self.num_rows:
                current_node = vertlist[nodes]
                queue = []
                id = current_node.get_id()
                previous_id = id
                current_id = current_node.list_of_nbrs[0]
                # = key_Set.dict_keys[0]
                queue.append(current_id)  # insert first node in the adj list
                while (current_id != id):
                    successor = vertlist[current_id].get_clockwise_sucessor(previous_id)
                    previous_id = current_id
                    current_id = successor
                    queue.append(current_id)

                face = frozenset(queue)
                faces_set.add(face)

        return faces_set

    def get_node_id(self, coordinates):
        id = coordinates[0] * self.num_columns + coordinates[1]
        return id


class node:
    __slots__ = ['i', 'j', 'id', 'adj_list', 'list_of_nbrs']

    def __init__(self, coordinates, num_columns):
        self.i = coordinates[0]
        self.j = coordinates[1]
        self.id = coordinates[0] * num_columns + coordinates[1]
        self.adj_list = {}
        self.list_of_nbrs = []

    def add_nbr(self, nbr, weight=1):
        self.adj_list[nbr] = weight
        self.list_of_nbrs.append(nbr.get_id())

    def get_coordinates(self):
        return self.i, self.j

    def get_next_nbr(self):
        pass

    def get_nbrs(self):
        return self.adj_list.keys()

    def get_id(self):
        return self.id

    def get_clockwise_sucessor(self, previous_id):
        previous_id_index = self.list_of_nbrs.index(previous_id)
        if previous_id_index == 0:
            return self.list_of_nbrs[len(self.list_of_nbrs) - 1]
        else:
            return self.list_of_nbrs[previous_id_index - 1]

    def __str__(self):
        rt_str = ""
        for keys in self.adj_list:
            rt_str += str(keys.get_id()) + " " + str(self.adj_list[keys]) + "\n"

        return rt_str


class simple_graph:
    __slots__ = ['vert_list', 'num_vertices']

    def __init__(self):
        self.vert_list = {}
        self.num_vertices = 0

    def add_simple_node(self, id):
        """
        :param key:  here the key will be the i and j coordinate of the nodes
        :return:
        """
        if id not in self.vert_list:
            new_node = simple_node(id)
            self.vert_list[id] = new_node
            self.num_vertices += 1

    def add_simple_edge(self, node1_id, node2_id, cost=1):
        """
        the definition might change based on the later requirements
        :param node1:
        :param node2:
        :param cost:
        :return:
        """
        node1 = self.vert_list[node1_id]
        node2 = self.vert_list[node2_id]
        node1.add_simple_nbr(node2, cost)

    def get_simple_node(self, id):
        return self.vert_list[id]

    def contract_edges(self, edge_set):
        """

        :param edge_set: The set of edges that needs to be contracted
        :return:
        """
        xnode_id = 'XNode'
        self.add_simple_node(xnode_id)

        x = self.get_simple_node(xnode_id)
        removed_edges = set()
        removed_nodes = set()

        for edges in edge_set:
            contract_node1_id = edges[0]
            contract_node2_id = edges[1]

            removed_nodes.add(edges[0])
            removed_nodes.add(edges[1])

            contract_node1 = self.get_simple_node(contract_node1_id)
            contract_node2 = self.get_simple_node(contract_node2_id)

            for nbr, wts in contract_node1.adj_list.items():
                if contract_node2_id != nbr.get_id():
                    if ((contract_node1_id, nbr.get_id()) not in edge_set \
                            and (nbr.get_id(), contract_node1_id) not in edge_set) and \
                            ((contract_node1_id,nbr.get_id()) not in removed_edges \
                              and (nbr.get_id(), contract_node1_id) not in removed_edges):

                        removed_edges.add((contract_node1_id,nbr.get_id()))
                        # print("removing edge",contract_node1_id,nbr.get_id())
                        x.add_simple_nbr(nbr, wts)
                        if nbr.get_id() != x.get_id():
                            nbr.add_simple_nbr(x, wts)
                        try:
                            nbr.remove_nbr(contract_node1)
                        except:
                            print("Exception:", contract_node1_id)

            for nbr, wts in contract_node2.adj_list.items():
                if contract_node1_id != nbr.get_id():
                    if ((contract_node2_id, nbr.get_id()) not in edge_set
                            and (nbr.get_id(), contract_node2_id) not in edge_set) and \
                            ((contract_node2_id, nbr.get_id()) not in removed_edges
                             and (nbr.get_id(), contract_node2_id) not in removed_edges):

                        removed_edges.add((contract_node2_id, nbr.get_id()))
                        # print("removing edge", contract_node2_id, nbr.get_id())
                        x.add_simple_nbr(nbr, wts)
                        if nbr.get_id() != x.get_id():
                            nbr.add_simple_nbr(x, wts)
                        try:
                            nbr.remove_nbr(contract_node2)
                        except:
                            print("Exception:",contract_node2_id)

        for node_id in removed_nodes:
            self.remove_node(node_id)

    def remove_node(self, id):
        del self.vert_list[id]

class simple_node:
    __slots__ = ['id', 'adj_list', 'list_of_nbrs']

    def __init__(self, id):
        self.id = id
        self.adj_list = {}

    def add_simple_nbr(self, nbr, weight=1):
        if nbr in self.adj_list:
            s = str(type(self.adj_list[nbr]))
            if s == '<class \'list\'>':
                if str(type(weight)) == '<class \'list\'>':
                    self.adj_list[nbr].extend(weight)
                else:
                    self.adj_list[nbr].append(weight)
            else:
                temp_wt = self.adj_list[nbr]
                self.adj_list[nbr] = []
                self.adj_list[nbr].append(weight)
                self.adj_list[nbr].append(temp_wt)
        else:
            self.adj_list[nbr] = weight

    def get_nbrs(self):
        return self.adj_list.keys()

    def get_id(self):
        return self.id

    def __str__(self):
        rt_str = ""
        for keys in self.adj_list:
            rt_str += str(keys.get_id()) + " " + str(self.adj_list[keys]) + "\n"

        return rt_str

    def remove_nbr(self,nbr):
        """
         removes a neighbor from the adjacency list
        :param nbr:  nbr is an object of class simple_node
        :return:
        """
        del self.adj_list[nbr]
