import numpy as np
#from scipy import interpolate
#import matplotlib.pyplot as plt
import cv2


class CatmullRomSpline():
    def __init__(self, pts):
        self.power = 0.5

        self.np_pts = np.zeros((len(pts) + 2, len(pts[0])))

        self.np_pts[1:-1, :] = np.array(pts)
        self.np_pts[0, :] = self.np_pts[1, :]
        self.np_pts[-1, :] = self.np_pts[-2, :]

        print(self.np_pts)

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

#new_pts = np.random.uniform(0, 1, (5,2))

new_pts = np.array([[0.0, 0.0, 10.0],
                    [100.0, 200.0, 15.0],
                    [130.0, 50.0, 10.0],
                    [255.0, 255.0, 20.0]])

sp = CatmullRomSpline(new_pts)

width = 256
canvas = np.zeros([width * 2, width * 2]).astype('float32')
for i in range(len(new_pts)-1):
    for j in range(100):
        pt = sp.getValue(i, j / 100.0)
        print(pt)
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), int(pt[2]), 1, -1)

cv2.imshow("VIZ", canvas)
cv2.waitKey(0)