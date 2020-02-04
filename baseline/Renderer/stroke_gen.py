import cv2
import numpy as np

class CatmullRomSpline():
    def __init__(self, pts):
        self.power = 0.5

        self.np_pts = np.zeros((len(pts) + 2, len(pts[0])))

        self.np_pts[1:-1, :] = np.array(pts)
        self.np_pts[0, :] = self.np_pts[1, :]
        self.np_pts[-1, :] = self.np_pts[-2, :]

        #print(self.np_pts)

    def calcVal(self, x0, x1, y0, y1, t):
        return (2.0 * x0 - 2.0 * x1 + y0 + y1) * t * t * t + (-3.0 * x0 + 3.0 * x1 - 2.0 * y0 - y1) * t * t + y0 * t + x0;

    def getValue(self, idx, t):
        p1 = self.np_pts[idx]
        p2 = self.np_pts[idx + 1]
        p3 = self.np_pts[idx + 2]
        p4 = self.np_pts[idx + 3]

        v0 = (p3 - p1) * self.power
        v1 = (p4 - p2) * self.power

        x = self.calcVal(p2[0], p3[0], v0[0], v1[0], t)
        y = self.calcVal(p2[1], p3[1], v0[1], v1[1], t)
        z = self.calcVal(p2[2], p3[2], v0[2], v1[2], t)

        return [x, y, z]

def normal(x, width):
    return (x * (width - 1) + 0.5)

def draw(pts, width=128):
    canvas = np.zeros([width * 2, width * 2]).astype('float32')

    pts = np.array(pts)
    pts = pts.reshape((-1, 3))

    pts[:, 0:-1] = normal(pts[:, 0:-1], width * 2)
    pts[:, -1] = normal(pts[:, -1], width // 4)
    #x0 = normal(x0, width * 2)
    #x1 = normal(x1, width * 2)
    #x2 = normal(x2, width * 2)
    #y0 = normal(y0, width * 2)
    #y1 = normal(y1, width * 2)
    #y2 = normal(y2, width * 2)
    #z0 = (int)(1 + z0 * width // 2)
    #z1 = (int)(1 + z1 * width // 2)
    #z2 = (int)(1 + z2 * width // 2)

    
    sp = CatmullRomSpline(pts)
    for i in range(len(pts)-1):
        for j in range(100):
            pt = sp.getValue(i, j / 100.0)
            #print(pt)
            if pt[2] < 0:
                pt[2] = 0
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), int(pt[2]), 1, -1)

    #for i in range(0,10,2):
    #    p = i*3
    #    f = (fs[p+0],fs[p+1],fs[p+2],fs[p+3],fs[p+4],fs[p+5],fs[p+6],fs[p+7],fs[p+8])
    #    drawOne(f, canvas, width)
    #print(canvas.shape)
    return 1 - cv2.resize(canvas, dsize=(width, width))

def drawOne(f, canvas, width):
    # x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x0, y0, x1, y1, x2, y2, z0, z1, z2 = f
    x1 = x0 + (x2 - x0) * x1

    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z1 = (int)(1 + z1 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * (1-t) * z0 + 2 * t * (1-t) * z1 + t * t * z2)
        # w = (1-t) * w0 + t * w2
        w = 1.0
        cv2.circle(canvas, (y, x), z, w, -1)
