import cv2
import numpy as np
image_URL = "C:\\Users\\ktjos\\Desktop\\Lumpy.jpg"

sink_vert = []

source_vert = []


def run_image_segmentation(image):
    pass


def get_clicked_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if param == 'source':
            source_vert.append((x,y))
        else:
            sink_vert.append((x,y))


def get_image():
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

if __name__ == '__main__':
    # run_image_segmentation()
    get_image()