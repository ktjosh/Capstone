import cv2
import numpy as np
image_URL = "C:\\Users\\ktjos\\Desktop\\Lumpy.jpg"

def run_image_segmentation(image):
    pass


def get_clicked_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(x, " ", y)


def get_image():
    img = cv2.imread(image_URL)
    height_img = img.shape[0]
    width_img = img.shape[1]

    cv2.namedWindow('my_window',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('my_window', width_img, height_img)
    cv2.setMouseCallback('my_window', get_clicked_points, param='Hello')

    cv2.imshow('my_window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # run_image_segmentation()
    get_image()