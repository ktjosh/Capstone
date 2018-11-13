from queue import Queue

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
        def give_elements(num_set):
            temp_array = []
            for nums in num_set:
                temp_array.append(nums)
            return temp_array

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
                        common_nodes = give_elements(common)
                        node1 = self.vert_list[common_nodes[0]]
                        node2 = self.vert_list[common_nodes[2]]
                        dual.add_simple_edge(vert1, vert2, node1.get_edge_wt(node2))
                        # dual.add_simple_edge(vert2, vert1)

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
        self.num_vertices += 1

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

        # remove self loops
        if x in x.adj_list:
            del x.adj_list[x]

    def remove_node(self, id):
        del self.vert_list[id]
        self.num_vertices -= 1

    def convert_to_list(self):
        for nodes in self.vert_list:
            self.vert_list[nodes].make_nbr_list()

    def _find_min_edge(self, edges):
        min_wt = float("Inf")
        min_edge = ""
        for edge, wts in edges.items():

            if wts<min_wt:
                min_wt = wts
                min_edge = edge

        return min_wt,min_edge

    def min_cut(self, source, sink):
        # first we will have the source and the sink vertices as the input
        # we will find each augmented path from source to sink
        # we will reverse the edges in the residual graph
        # all the edges reachable node from source  to the non reachable nodes are min cut
        # the BFS should return boolean if there exist a path
        # backtrack the path from sink to source along with edge wt
        # keep extra information of edge wt
        parent = {}
        edge = {}
        min_cut_edges = set()
        source_set_edges = set()

        while self.BFS(parent, edge, source, sink):
            # print("E:", edge)
            # print("P:", parent)
            min_wt, min_edge = self._find_min_edge(edge)
            min_cut_edges.add(min_edge)

            current_node = sink

            isSourceSink_node = False

            while current_node!= source:
                parent_node = parent[current_node]

                if parent_node == min_edge[0]:
                    isSourceSink_node = True

                if isSourceSink_node:
                    source_set_edges.add(parent_node)
                current_node.add_wt(parent_node, min_wt, edge[(parent_node, current_node)])
                parent_node.subtract_wt(current_node, min_wt, edge[(parent_node, current_node)])

                current_node = parent_node

            parent = {}
            edge = {}

        # source_set_edges = self._BFS_Traversal(source)
        return min_cut_edges,source_set_edges

    def BFS(self, parent, edge, source, sink):
        # edge must be put as a tuple with convention (parent, children)
        # source and sink should be the objects of source and sink
        # print("In BFS","source:", source,"Sink:", sink)
        queue = Queue()
        visited = set()

        queue.put(source)
        found_path = False
        while not queue.empty():
            current_node = queue.get()
            visited.add(current_node)

            for nbrs in current_node.adj_list.keys():
                if nbrs not in visited:
                    queue.put(nbrs)
                    parent[nbrs] = current_node
                    visited.add(nbrs)
                    if nbrs == sink:
                        found_path = True
                        break

            if found_path:
                break

        if found_path:
            # track the path backwords and add edges
            current_path_node = sink

            while current_path_node!= source:
                parent_node = parent[current_path_node]
                edge_wt = parent_node._get_edge_wt( current_path_node)
                edge[(parent_node,current_path_node)] = edge_wt
                current_path_node = parent_node

        return found_path

    def _BFS_Traversal(self,source):
        queue = Queue()
        visited = set()

        queue.put(source)
        while not queue.empty():
            current_node = queue.get()
            visited.add(current_node)

            for nbrs in current_node.adj_list.keys():
                if nbrs not in visited:
                    queue.put(nbrs)
                    visited.add(nbrs)

        return visited
class simple_node:
    __slots__ = ['id', 'adj_list', 'list_of_nbrs']

    def __init__(self, id):
        self.id = id
        self.adj_list = {}

    def add_simple_nbr(self, nbr, weight=1):
        """

        :param nbr: object of class node
        :param weight:
        :return: None
        """
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

    def make_nbr_list(self):
        for nbr in self.adj_list:
            if str(type(self.adj_list[nbr])) != '<class \'list\'>':
                temp = self.adj_list[nbr]
                self.adj_list[nbr] = []
                self.adj_list[nbr].append(temp)

    def add_wt(self, parent_node, min_wt, edge_wt_considered):
        # adding the new reverse edge of the edge wt towards the parent node
        self.adj_list[parent_node].append(min_wt)

    def subtract_wt(self, child_node, min_wt, edge_wt_considered):
        """
        if edge_list becomes empty remove the node
        we will be dealing with float edge wts hence we may not find the exact edge wt
        but most probably we will.

        Add a threshold to edge wt, if the edge wt is below 0.000009 then remove that edge
        :param child_node:
        :param min_wt:
        :param edge_wt_considered:
        :return:
        """
        self.adj_list[child_node].remove(edge_wt_considered)
        threshold = 0.0009
        if min_wt == edge_wt_considered:
            if len(self.adj_list[child_node]) == 0:
                del self.adj_list[child_node]
        else:
            new_wt = edge_wt_considered - min_wt
            if new_wt > threshold:
                self.adj_list[child_node].append(new_wt)

    def _get_edge_wt(self, child):
        # largest_non_zero_edge
        edge_wt = max(self.adj_list[child])
        return edge_wt
