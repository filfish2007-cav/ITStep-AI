import math

def get_angle(x1, y1, x2, y2, x3, y3):
    # x2,y2 - центральна точка (коліно)
    a1 = math.atan2(y1 - y2, x1 - x2)
    a2 = math.atan2(y3 - y2, x3 - x2)
    angle = math.degrees(a1 - a2)
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle