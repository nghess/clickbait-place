import clr
clr.AddReference("OpenCV.Net")
clr.AddReference("System")
from OpenCV.Net import *

centroid_radius = 25
centroid_color = Scalar.Rgb(255, 0, 0)

@returns(IplImage)
def process(value):
    centroid_x = value.Item1
    centroid_y = value.Item2
    img = value.Item3

    CV.Circle(img, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, 1)

    return img
