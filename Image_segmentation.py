import math
import cv2
import numpy as np
image_URL = "C:\\Users\\ktjos\\Desktop\\Lumpy.jpg"

sink_vert = []
source_vert = []


def run_image_segmentation(image):
    pass



def pre_process_image():
    """
    Later might want to add some preprocessing which will involve smoothening of the image
    :return:
    """
    img = get_image(image_URL)

    # convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    diff = abs(luminance1 - luminance2)
    wt = 1 / math.pow(diff, 2)
    return wt

def edge_wt_function2(luminance1, luminance2):
    diff = abs(luminance1 - luminance2)
    wt = math.pow((255 - diff), 8)
    return wt

def edge_wt_function3(luminance1, luminance2):
    diff = abs(luminance1 - luminance2)
    wt = 1000/math.pow(math.e,diff)
    return wt


if __name__ == '__main__':
    # run_image_segmentation()
    get_image()