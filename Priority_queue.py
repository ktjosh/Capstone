


def heap_function(entity1, entity2):
    # first element smaller than second
    # considering that both entities are a tupple and we have to
    # check the 2nd entry in each tuple, i.e value at index 1
    return entity1[1]<entity2[1]
    # return entity1 < entity2

class PriorityQueue:
    """

    """
    __slots__ = ['list','size','heap_condition']

    def __init__(self):
        self.list = []
        self.size = 0
        self.heap_condition = heap_function


    def parent(self,index):
        return (index-1)//2


    def insert(self,data):
        self.list.append(data)
        self.size += 1
        self._bubbleup(self.size-1)


    def pop(self):
        pop_elem = self.list[0]
        self.size -= 1
        if len(self.list)-1 > 0:
            self.list[0] = self.list.pop()
            self._bubbledown(0)
        return pop_elem


    def _bubbleup(self, bubbleup_index):
        while bubbleup_index > 0 and heap_function(self.list[bubbleup_index], self.list[self.parent(bubbleup_index)]):
            self.swap(bubbleup_index, self.parent(bubbleup_index))
            bubbleup_index = self.parent(bubbleup_index)


    def _bubbledown(self, index):

        swap_index = self.smallest_index(index)

        while index!= swap_index:
            self.swap(index,swap_index)
            index = swap_index
            swap_index = self.smallest_index(index)


    def swap(self, index1, index2):
        temp = self.list[index1]
        self.list[index1] = self.list[index2]
        self.list[index2] = temp

    def smallest_index(self, index):

        child1 = (index*2) +1
        child2 = (index*2) +2
        if child1 >= self.size:
            # meaning it doesnt exist
            return index
        if child2 >= self.size:
            # possibility that child1 exist
            if self.heap_condition(self.list[child1], self.list[index]):
                return child1
            else:
                return index

        # three way comparision
        if self.heap_condition(self.list[child1], self.list[child2]):
            if self.heap_condition(self.list[child1], self.list[index]):
                return child1
            else:
                return index
        else:
            if self.heap_condition(self.list[child2], self.list[index]):
                return child2
            else:
                return index



