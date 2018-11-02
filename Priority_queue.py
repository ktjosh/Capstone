class PriorityQueue:
    """

    """
    __slots__ = ['list', 'size', 'dist_dict', 'positions']

    def __init__(self, dist_dict):
        self.list = []
        self.size = 0
        self.dist_dict = dist_dict
        self.positions = {}

    def parent(self, index):
        return (index - 1) // 2

    def insert(self, data):
        self.list.append(data)
        self.positions[data] = self.size
        self.size += 1
        self.bubbleup(self.size - 1)

    def pop(self):
        pop_elem = self.list[0]
        # remove the element from the dictionary
        del self.positions[pop_elem]
        self.size -= 1
        if len(self.list) - 1 > 0:
            # put the last element to the root of the queue
            self.list[0] = self.list.pop()
            # change the position of the top element
            self.positions[self.list[0]] = 0
            self.bubbledown(0)
        return pop_elem

    def bubbleup(self, bubbleup_index):
        while bubbleup_index > 0 and self.heap_function(self.list[bubbleup_index],
                                                        self.list[self.parent(bubbleup_index)]):
            self.swap(bubbleup_index, self.parent(bubbleup_index))
            bubbleup_index = self.parent(bubbleup_index)

    def bubbledown(self, index):

        swap_index = self.smallest_index(index)

        while index != swap_index:
            self.swap(index, swap_index)
            index = swap_index
            swap_index = self.smallest_index(index)

    def swap(self, index1, index2):
        temp = self.list[index1]
        self.list[index1] = self.list[index2]
        self.list[index2] = temp

        # chaning the index in position
        self.positions[self.list[index1]] = index1
        self.positions[self.list[index2]] = index2


    def smallest_index(self, index):

        child1 = (index * 2) + 1
        child2 = (index * 2) + 2
        if child1 >= self.size:
            # meaning it doesnt exist
            return index
        if child2 >= self.size:
            # possibility that child1 exist
            if self.heap_function(self.list[child1], self.list[index]):
                return child1
            else:
                return index

        # three way comparision
        if self.heap_function(self.list[child1], self.list[child2]):
            if self.heap_function(self.list[child1], self.list[index]):
                return child1
            else:
                return index
        else:
            if self.heap_function(self.list[child2], self.list[index]):
                return child2
            else:
                return index

    def update_priority(self,element):
        update_element_index = self.positions[element]
        self.bubbleup(update_element_index)

    def heap_function(self, entity1, entity2):
        # first element smaller than second
        # considering that both entities are a tupple and we have to
        # check the 2nd entry in each tuple, i.e value at index 1
        return self.dist_dict[entity1] < self.dist_dict[entity2]
        # return entity1 < entity2
