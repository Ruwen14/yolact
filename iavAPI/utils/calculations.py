import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from data import COLORS
import random
import math
import torch


def calc_rotated_bbox(cntrs, mode=None):
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


def getOrientationAxis(center, angle,height, width, scale):
    axis_major = (max(height, width) / 2)*scale
    axis_minor =  (min(height, width) / 2)*scale
    xc, yc = center

    # Longer Axis
    xtop = xc + math.cos(math.radians(angle)) * axis_major
    ytop = yc + math.sin(math.radians(angle)) * axis_major
    xbot = xc + math.cos(math.radians(angle + 180)) * axis_major
    ybot = yc + math.sin(math.radians(angle + 180)) * axis_major

    # Shorter Axis
    xleft = xc + math.cos(math.radians(angle+90)) * axis_minor
    yleft = yc + math.sin(math.radians(angle+90)) * axis_minor
    xright= xc + math.cos(math.radians(angle - 90)) * axis_minor
    yright = yc + math.sin(math.radians(angle - 90)) * axis_minor

    line_long = ((int(xtop), int(ytop)), (int(xbot), int(ybot)))
    line_short = ((int(xleft), int(yleft)), (int(xright), int(yright)))


    return line_long, line_short

def getTextTag(position, text_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, fontThickness=1):
    text_w, text_h = cv2.getTextSize( text_str, fontFace, fontScale, fontThickness)[0]
    text_p = (position[0], position[1])

    return text_p, text_w, text_h



def fit_ellipse(cntrs):
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


def calc_rotated_bbox_full(binary_image, slice: bool=True, contour_mode=cv2.CHAIN_APPROX_NONE):
    cntrs = getContours(binary_image, contour_mode)

    if cntrs is not None:
        if slice:
            try: # ToDo Slice step canot be zero --> need to investigate further
                cntrs = sliceNSortContours(cntrs, 30, sort_mode=True)
            except:
                print("Skipped Frame")

        box = [np.int0(cv2.boxPoints(cv2.minAreaRect(cntr)))  for cntr in cntrs]

        return box

    return None


def fix_color(number_det, colors):
    """
    Temporary Fix for rotated Bounding Boxex to have same color like text
    """
    number_colors = len(colors)
    if number_det > number_colors:
        diff = number_det - number_colors
        [colors.append(random.choice(COLORS)) for i in range(diff)]
        return colors

    elif number_det < number_colors:
        diff  =number_colors - number_det
        return colors[:-diff] #Shorten by ne

    else:
        return colors



def order_rect(pts):
    #Aus:https://github.com/jrosebr1/imutils/blob/df65a77087548b5952431cc8006d66bcca7b1abc/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-left, and bottom-right order
    return np.array([tl, tr, bl, br])


def find_rect_centroid(top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    xCenter = int((x1 + x2) / 2)
    yCenter = int((y1 + y2) / 2)

    return(xCenter,yCenter)


def calc_min_dist(A, B):
    distance = cdist(A, B)
    rel_dist_minima = distance.min(axis=1)
    abs_dist_min = rel_dist_minima.min()
    abs_dist_min_idx = np.argmin(rel_dist_minima)
    min_idx = np.argmin(distance, axis=1)[abs_dist_min_idx]

    return (A[abs_dist_min_idx], B[min_idx]), abs_dist_min

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

def sliceNSortContours(cntrs, slice_stride:int, sort_mode:bool =False):
    """
    Args:
        cntrs: cntrs generated by calculations.getContours
        slice_stride: every N'th entry will remain
        sort_mode: bool, Whether to sort Contours from smallest to highest entry

    Returns: sliced or/and sorted contours
    """


    cnt_sliced = [cnt[1::math.ceil(len(cnt) / slice_stride)].reshape(-1, 2) for cnt in cntrs]


    if sort_mode == False:
       return cnt_sliced
    else:
        cnt_sorted = [cnt_s[np.argsort(cnt_s[:, 0])] for cnt_s in cnt_sliced]
        cnt_sorted.sort(key=lambda x: x[0][0])
        return cnt_sorted

def closest_node(node, nodes):
    closest_index = cdist(node, nodes).argmin()
    return nodes[closest_index]



def pytorch_cos_sim(a, b):
    import torch

    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def find_closest_points(PointsA, PointsB, mode='GPU'):
    print(torch.cuda.get_device_name(0))
    if mode == 'CPU':
        MATRIX_distance = cdist(PointsA, PointsB)
        VECTOR_min_distance_rel= MATRIX_distance.min(axis=1)
        FLOAT_min_distance_abs = VECTOR_min_distance_rel.min()
        INT_min_distance_abs_idx = np.argmin(VECTOR_min_distance_rel)

        closestA = tuple(PointsA[INT_min_distance_abs_idx])
        closestB =  tuple(PointsB[np.argmin(MATRIX_distance, axis=1)[INT_min_distance_abs_idx]])
        closest_Points = (closestA, closestB)
        return closest_Points, FLOAT_min_distance_abs

    if mode == 'GPU':
        assert torch.cuda.current_device() == 0, 'Cant use "mode==GPU" without GPU'

        TENSOR_A = torch.FloatTensor(PointsA)
        TENSOR_B = torch.FloatTensor(PointsB)
        TENSOR_Distance = torch.cdist(TENSOR_A, TENSOR_B)
        mins, indices = torch.min(TENSOR_Distance, dim=1)
        TENSOR_min_distance_abs = torch.min(mins)

        TENSOR_closestA = TENSOR_A[torch.argmin(mins)]
        TENSOR_closestB = TENSOR_B[torch.argmin(TENSOR_Distance, dim=1)[torch.argmin(mins)]]

        ARRAY_closestA = TENSOR_closestA.cpu().detach().numpy()
        ARRAY_closestB = TENSOR_closestB.cpu().detach().numpy()
        closest_Points = (tuple(ARRAY_closestA.astype(int)), tuple(ARRAY_closestB.astype(int)))
        FLOAT32_min_distance_abs = TENSOR_min_distance_abs.cpu().detach().numpy()

        return closest_Points, float(FLOAT32_min_distance_abs)




def pointwise_distance(PointsA, PointsB, mode='GPU'):
    print(torch.cuda.get_device_name(0))
    if mode == 'CPU':
        distance = cdist(PointsA, PointsB)
        return distance

    if mode == 'GPU':
        Tensor_A = torch.FloatTensor(PointsA)
        Tensor_B = torch.FloatTensor(PointsB)
        distance = torch.cdist(Tensor_A, Tensor_B)
        distance_on_cpu = distance.cpu().detach().numpy()
        return distance_on_cpu
    else:
        print("choose mode")
        exit()


def convex_hull_image(data):
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v,0], region[v,1]) for v in hull.vertices]
    img = Image.new('L', data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T

def approx_line(img=None,cntrs=None):
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cntrs[0], cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)


def sort_by_distance(ListA, ListB, returnType='List'):
    """

    Args:
        ListA:
        ListB:

    Returns:
        sorted ListB, so that each entry x --> ListB[x] has least distance to entry x of ListA[x]
    """
    distance_matrix = cdist(ListA, ListB)

    minima_indices = [np.argmin(vector) for vector in distance_matrix]

    if returnType == 'List':
        # Sort ListB by Indice of Minima
        sorted_ListB = [ListB[indice] for indice in minima_indices]
        return sorted_ListB

    else:
        return minima_indices







def cdist_loop(a, b):
    distance = [cdist(a, b[i]) for i in range(len(b))]
    return distance



def pointwise_distance_VEC2MAT(VECTOR_A, MATRIX_B, mode='CPU'):
    if not (mode=='GPU' or mode=='CPU' or mode=='numba' or mode=='tree'):
        mode='CPU'; print("Force mode -> CPU")

    # Slow! 566ms @12500 points, 1.88 s @12500000 points
    if mode =='numba':
        from numba import jit # will be cached no worries

        @jit(nopython=True, fastmath=True)
        def euclidean(u, v):
            n = len(v)
            dist = np.zeros((n))
            for i in range(n):
                dist[i] = np.sqrt(np.nansum((u - v[i]) ** 2))
            return dist

        @jit(nopython=True, fastmath=True)
        def vector_to_matrix_distance(u, m):
            m_rows = m.shape[0]
            m_cols = m.shape[1]
            u_rows = u.shape[0]

            out_matrix = np.zeros((m_rows, u_rows, m_cols))
            for i in range(m_rows):
                for j in range(u_rows):
                    out_matrix[i][j] = euclidean(u[j], m[i])

            return out_matrix


        return vector_to_matrix_distance(VECTOR_A,MATRIX_B)

    # 119 μs @12500 points
    # 96 ms @12500000 points
    if mode =='CPU':
        # Fast cdist function
        # List compression for speed up
        distance_matrix = np.array([cdist(VECTOR_A  , MATRIX_B[i])
                                    for i in range(len(MATRIX_B))])
        return distance_matrix

    # 351 μs @12500 points
    # 47 ms @12500000 points
    if mode =='GPU':
        Tensor_A = torch.FloatTensor(VECTOR_A)
        Tensor_B = torch.FloatTensor(MATRIX_B)

        distance_matrix =np.array([torch.cdist(Tensor_A, Tensor_B[i]).cpu().detach().numpy()
                                    for i in range(len(Tensor_B))])

        return distance_matrix

    if mode =='tree':
        dist = np.array([cKDTree(vec).query(VECTOR_A) for vec in (MATRIX_B)])

        return dist







