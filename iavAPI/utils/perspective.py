import cv2
import math


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


def fitTextTag(position, text_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, fontThickness=1):
    text_w, text_h = cv2.getTextSize( text_str, fontFace, fontScale, fontThickness)[0]
    text_p = (position[0], position[1])

    return text_p, text_w, text_h