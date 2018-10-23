from Node import *

def create_graph():
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
                grph.add_edge((i,j),(i,j+1),matrix[i][j])
            if i + 1 < len(matrix):
                grph.add_edge((i,j),(i+1,j),matrix[i][j])
            if j -1 >= 0:
                grph.add_edge((i,j),(i,j-1),matrix[i][j])
            if i -1 >=0:
                grph.add_edge((i,j),(i-1,j),matrix[i][j])

    # for keys in grph.vert_list.keys():
    #     print(grph.vert_list[keys])
    #
    # for keys in grph.vert_list.keys():
    #     print(grph.vert_list[keys].list_nbrs)

    dual = grph.get_dual()

    for id,node in dual.vert_list.items():
        print("**", id ," " ,node)

    print(grph.find_faces(grph.vert_list))



def tp():
    v = {}

    for i in range(5):
        v[i] = i**3

    for i in v.keys():
        print(i)


if __name__ == '__main__':
    create_graph()
    # tp()