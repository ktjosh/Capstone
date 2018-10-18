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
            new_node = node(coordinates,self.num_columns)
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


    def get_dual(self, vertlist):
        """
        function finds dual graph of a given graph
        :param vertlist:
        :return:
        """

        dual_vertices = self.find_faces(vertlist)

    def find_faces(self, vertlist):
        faces_set = set()

        for nodes in vertlist.keys():
            (i, j) = vertlist[nodes].get_coordinates()

            if i+1 != self.num_rows:
                current_node = vertlist[nodes]
                queue = []
                id = current_node.get_id()
                previous_id = id
                current_id = current_node.list_of_nbrs[0]
                 # = key_Set.dict_keys[0]
                queue.append(current_id) # insert first node in the adj list
                while(current_id!=id):
                    successor = vertlist[current_id].get_clockwise_sucessor(previous_id)
                    previous_id = current_id
                    current_id = successor
                    queue.append(current_id)

                face = frozenset(queue)
                faces_set.add(face)

        return faces_set







    def get_node_id(self,coordinates):
        id = coordinates[0]*self.num_columns + coordinates[1]
        return id



class node:
    __slots__ = ['i', 'j', 'id', 'adj_list', 'list_of_nbrs']

    def __init__(self, coordinates, num_columns):
        self.i = coordinates[0]
        self.j = coordinates[1]
        self.id = coordinates[0]*num_columns + coordinates[1]
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
            return self.list_of_nbrs[len(self.list_of_nbrs)-1]
        else:
            return self.list_of_nbrs[previous_id_index -1]


    def __str__(self):
        rt_str = ""
        for keys in self.adj_list:
            rt_str += str(keys.get_id()) + " " + str(self.adj_list[keys]) + "\n"

        return rt_str