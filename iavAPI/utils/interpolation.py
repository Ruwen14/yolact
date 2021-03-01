import cv2
import numpy as np
import math

def calcRotatedBBox(cntrs, mode=None):
    rect = cv2.minAreaRect(cntrs)
    box = np.int0(cv2.boxPoints(rect))
    if mode == 'only_box':
        return box

    else:
        center = (int(rect[0][0]), int(rect[0][1]))
        (width, height), angle = rect[1:3]

        if width < height:
            angle = angle + 90
        else:
            angle = angle + 180

        return box, center, angle, width, height

def calcBBoxCentroid(top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    xCenter = int((x1 + x2) / 2)
    yCenter = int((y1 + y2) / 2)

    return(xCenter,yCenter)



def getContours(img_array, mode=cv2.CHAIN_APPROX_NONE):
    """
    Args:
        img_array: nd.array of image
        mode: cv2.CHAIN_APPROX_NONE OR cv2.CHAIN_APPROX_SIMPLE

    Returns: cntrs of image
    """
    im_bw = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_bw, 128, 255, 0)

    cntrs, _ = cv2.findContours(im_bw, cv2.RETR_TREE, mode)
    return cntrs


def fitEllipse(cntrs):
    ellipse = cv2.fitEllipse(cntrs)
    (xc,yc),(d1,d2),angle = ellipse

    # draw circle at center
    xc, yc = ellipse[0]

    # draw vertical line
    # compute major radius
    rmajor = max(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    xtop = xc + math.cos(math.radians(angle))*rmajor
    ytop = yc + math.sin(math.radians(angle))*rmajor
    xbot = xc + math.cos(math.radians(angle+180))*rmajor
    ybot = yc + math.sin(math.radians(angle+180))*rmajor



def approxLine(img=None,cntrs=None):
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cntrs[0], cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

def imageConvexHull(data):
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v,0], region[v,1]) for v in hull.vertices]
    img = Image.new('L', data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T

