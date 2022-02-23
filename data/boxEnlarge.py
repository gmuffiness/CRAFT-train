import numpy as np
import math
import config.config


def enlarge_box_for_rectangle(box, imgh, imgw):
    boxw = box[1][0] - box[0][0]
    boxh = box[2][1] - box[1][1]

    if imgh <= boxh or imgw <= boxw:
        return box

    enlargew = boxw * 0.5
    enlargeh = boxh * 0.5

    # 상자는 명확성을 위해 이 부분을 확장하고 코드는 중복으로 작성됩니다.
    # 확장 후 왼쪽 상단 모서리 정점의 좌표, 나머지 점은 시계 방향 등입니다.
    box[0][0], box[0][1] = (
        max(0, box[0][0] - int(enlargew * 0.5)),
        max(0, box[0][1] - int(enlargeh * 0.5)),
    )
    box[1][0], box[1][1] = (
        min(imgw, box[1][0] + int(enlargew * 0.5)),
        max(0, box[1][1] - int(enlargeh * 0.5)),
    )
    box[2][0], box[2][1] = (
        min(imgw, box[2][0] + int(enlargew * 0.5)),
        min(imgh, box[2][1] + int(enlargeh * 0.5)),
    )
    box[3][0], box[3][1] = (
        max(0, box[3][0] - int(enlargew * 0.5)),
        min(imgh, box[3][1] + int(enlargeh * 0.5)),
    )

    return box

def pointAngle(Apoint, Bpoint):
    angle = (Bpoint[1] - Apoint[1]) / ((Bpoint[0] - Apoint[0]) + 10e-8)
    return angle

def pointDistance(Apoint, Bpoint):
    return math.sqrt((Bpoint[1] - Apoint[1])**2 + (Bpoint[0] - Apoint[0])**2)

def lineBiasAndK(Apoint, Bpoint):

    K = pointAngle(Apoint, Bpoint)
    B = Apoint[1] - K*Apoint[0]
    return K, B

def getX(K, B, Ypoint):
    return int((Ypoint-B)/K)

def sidePoint(Apoint, Bpoint, h, w, placehold):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    halfIncreaseDistance = config.config.ENLARGEBOX_MAGINE * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * halfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * halfIncreaseDistance)

    if placehold == 'leftTop':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

# 将box扩大1.5倍
def enlargebox(box, h, w):


    # box = [Apoint, Bpoint, Cpoint, Dpoint]
    Apoint, Bpoint, Cpoint, Dpoint = box
    K1, B1 = lineBiasAndK(box[0], box[2])
    K2, B2 = lineBiasAndK(box[3], box[1])
    X = (B2 - B1)/(K1 - K2)
    Y = K1 * X + B1
    center = [X, Y]

    x1, y1 = sidePoint(Apoint, center, h, w, 'leftTop')
    x2, y2 = sidePoint(center, Bpoint, h, w, 'rightTop')
    x3, y3 = sidePoint(center, Cpoint, h, w, 'rightBottom')
    x4, y4 = sidePoint(Dpoint, center, h, w, 'leftBottom')
    newcharbox = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return newcharbox
