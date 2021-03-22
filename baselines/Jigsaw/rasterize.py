import numpy as np
from bresenham import bresenham
import scipy.ndimage


def mydrawPNG(vector_image, Side = 256):

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    stroke_bbox = []
    stroke_cord_buffer = []
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)
        stroke_cord_buffer.extend([list(i) for i in cordList])

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        if vector_image[i, 2] == 1:
            min_x = np.array(stroke_cord_buffer)[:, 0].min()
            min_y = np.array(stroke_cord_buffer)[:, 1].min()
            max_x = np.array(stroke_cord_buffer)[:, 0].max()
            max_y = np.array(stroke_cord_buffer)[:, 1].max()
            stroke_bbox.append([min_x, min_y, max_x, max_y])
            stroke_cord_buffer = []

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    #utils.image_boxes(Image.fromarray(raster_image).convert('RGB'), stroke_bbox).show()
    return raster_image, stroke_bbox


def preprocess(sketch_points, side = 256.0):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([800, 800])
    sketch_points[:,:2] = sketch_points[:,:2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_Sketch(sketch_points):
    sketch_points = preprocess(sketch_points)
    raster_images, stroke_bbox = mydrawPNG(sketch_points)
    return raster_images, stroke_bbox