from Node import *
from dijkstra import *
import math
import cv2
import numpy as np
image_URL = "C:\\Users\\ktjos\\Desktop\\plane2.jpg"

sink_vert = []
source_vert = []


def run_image_segmentation(image):
    # source_vert = [(7, 19)]
    # sink_vert = [(24, 55), (35, 82), (73, 55)]
    # source_vert = [(7, 18)]
    # sink_vert = [(42, 25), (47, 46), (52, 53), (68, 29), (80, 41), (73, 44)]
    # source_vert = [(5, 4)]
    # sink_vert =[(32, 11), (21, 37), (16, 59), (5, 64), (8, 73), (25, 93), (62, 92), (82, 86), (87, 60), (92, 37), (79, 39),
    #  (67, 26), (54, 17)]
    # for plane 2
    # source_vert = [(50, 21)]
    # sink_vert = [(19, 48), (25, 47), (21, 44)]
    graph = create_graph(image, edge_wt_function1)
    d, source_nodes, sink_nodes = graph.get_dual(source_vert, sink_vert)
    print(graph.vert_list[0])

    print("::Running Dijkstra")
    idx = len(sink_nodes)//2
    start_node_id = sink_nodes[0]
    dist, parent = run_dijkstra(d, start_node_id)

    print("Dijkstra Complete")

    edge_set = set()
    prev = ""

    for sink_vertex in sink_nodes:
        current_vertex = sink_vertex
        prev = ""
        # print(current_vertex, " ", start_node_id)
        # print(current_vertex==start_node_id)
        while current_vertex!=start_node_id and current_vertex!=None:
            prev = parent[current_vertex]
            edge_set.add((prev, current_vertex))
            current_vertex = prev

    start_node_id = sink_nodes[idx]
    dist, parent = run_dijkstra(d, start_node_id)
    for sink_vertex in sink_nodes:
        current_vertex = sink_vertex
        prev = ""
        while current_vertex != start_node_id:
            prev = parent[current_vertex]
            edge_set.add((prev, current_vertex))
            current_vertex = prev
    # for vertice in dist.keys():
    #     print(vertice, " ", dist[vertice], " ", parent[vertice])
    #
    # print(edge_set)

    d.contract_edges(edge_set)
    d.convert_to_list()
    # print("***************************************************************")
    # for nodes in d.vert_list:
    #     print("-----------------------------------------------------------")
    #     print(d.vert_list[nodes])
    # print("***************************************************************")
    sink = d.get_simple_node('XNode')
    # print(sink)
    # print("-------------------------------------------------------")
    # print(d.get_simple_node(frozenset({728, 729, 828, 829})))
    # print("-------------------------------------------------------")
    #
    # print(d.get_simple_node(frozenset({729, 730, 829, 830})))
    # print("-------------------------------------------------------")

    source = d.get_simple_node(source_nodes[0])
    print("Source_v:",source)
    print("Sink_V:", sink)
    # min_cut_edges, source_set_edges = d.min_cut(source, sink)
    for nodes in source.adj_list.keys():
        a = source.adj_list[nodes]
        b =[]
        for it in a:
            b.append(it*1)
        source.adj_list[nodes] = b
    print("Source_v:", source)
    min_cut_edges, source_set_edges = d.min_cut(source, sink)
    print("Source_v:", source)

    print("number of mincut edges", len(min_cut_edges))
    # print(min_cut_edges)
    # for nodes in min_cut_edges:
    #     print('==========================')
    #     print(nodes[0].get_id(), "edges:", nodes[0])
    #     print(nodes[1].get_id(), "edges:", nodes[1])
    #     print('==========================')
    #
    # for nodes in source_set_edges:
    #     print('------------------------------')
    #     print(nodes.get_id(), "edges:", nodes)
    img = cv2.imread(image_URL)


    green = [0,255,0]
    for edges in min_cut_edges:
        try:
            print(edges[0].get_id()," ", edges[1].get_id())
            intersection = edges[0].get_id() & edges[1].get_id()
        except:
            continue
        cord = []
        for nodes in intersection:
            x,y = graph.get_coordinates(nodes)
            cord.append((x,y))
            img[x,y] = green

    cv2.namedWindow('Source', cv2.WINDOW_NORMAL)
    cv2.imshow('Source', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('C:\\Users\\ktjos\\Desktop\\snap_segmentation.jpg', img)
    # for nodes in min_cut_edges:




def create_graph(image, edge_wt_function):
    graph = Graph(len(image), len(image[0]))

    for i in range(len(image)):
        for j in range(len(image[0])):
            graph.add_node((i, j))

    for i in range(len(image)):
        for j in range(len(image[0])):
            # try for all four neighbors in order: right, bottom, left, top
            if j + 1 < len(image[0]):
                graph.add_edge((i, j), (i, j + 1), edge_wt_function(image[i][j], image[i][j + 1]))
            if i + 1 < len(image):
                graph.add_edge((i, j), (i + 1, j), edge_wt_function(image[i][j], image[i + 1][j]))
            if j - 1 >= 0:
                graph.add_edge((i, j), (i, j - 1), edge_wt_function(image[i][j], image[i][j - 1]))
            if i - 1 >= 0:
                graph.add_edge((i, j), (i - 1, j), edge_wt_function(image[i][j], image[i - 1][j]))


    return graph

def pre_process_image():
    """
    Later might want to add some preprocessing which will involve smoothening of the image
    :return:
    """
    img = get_image(image_URL)
    sink_vert = [(7, 19)]
    source_vert = [(24, 55), (35, 82), (73, 55)]
    # convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.blur(img, (5, 5))
    img_gray = [[0 for i in range(len(img[0]))] for i in range(len(img))]
    for i in range(len(img)):
        for j in range(len(img[0])):
            luminence = (0.2125 * int(img[i][j][0])) + (0.7152 * int(img[i][j][1])) + (0.0722 * int(img[i][j][2]))
            img_gray[i][j] = luminence
    run_image_segmentation(img_gray)

def get_clicked_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if param == 'source':
            source_vert.append((x,y))
        else:
            sink_vert.append((x,y))

def get_image(image_URL):
    img = cv2.imread(image_URL)
    height_img = img.shape[0]
    width_img = img.shape[1]

    cv2.namedWindow('Source',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Source', width_img, height_img)
    cv2.setMouseCallback('Source', get_clicked_points, param='source')

    cv2.imshow('Source', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow('Sink', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sink', width_img, height_img)
    cv2.setMouseCallback('Sink', get_clicked_points, param='sink')

    cv2.imshow('Sink', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(source_vert)
    print(sink_vert)

    return img

def edge_wt_function1(luminance1, luminance2):
    diff = (abs(int(luminance1) - int(luminance2)))
    wt = 1 / math.pow((diff+1), 2)
    return wt

def edge_wt_function2(luminance1, luminance2):
    diff = abs(int(luminance1) - int(luminance2))
    wt = math.pow((255 - diff), 2)
    return wt

def edge_wt_function3(luminance1, luminance2):
    diff = abs(int(luminance1) - int(luminance2))
    wt = 1000/math.pow(math.e,diff)
    return wt


if __name__ == '__main__':
    # run_image_segmentation()
    # get_image()
    pre_process_image()