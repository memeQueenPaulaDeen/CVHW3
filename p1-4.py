import numpy as np


def homography(mapping):
	#mapping -> (x,y):(xp,yp)
	pointList = list(mapping.keys())
	assert len(mapping) > 3
	xy1 = pointList[0]
	x1 = xy1[0]
	y1 = xy1[1]
	xy2 = pointList[1]
	x2 = xy2[0]
	y2 = xy2[1]
	xy3 = pointList[2]
	x3 = xy3[0]
	y3 = xy3[1]
	xy4 = pointList[3]
	x4 = xy4[0]
	y4 = xy4[1]

	x1p = mapping[xy1][0]
	y1p = mapping[xy1][1]
	x2p = mapping[xy2][0]
	y2p = mapping[xy2][1]
	x3p = mapping[xy3][0]
	y3p = mapping[xy3][1]
	x4p = mapping[xy4][0]
	y4p = mapping[xy4][1]
	p = [x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p]

	x = [[x1, y1, 1, 0, 0, 0, -x1p * x1, -x1p * y1],
	     [0, 0, 0, x1, y1, 1, -y1p * x1, -y1p * y1],
	     [x2, y2, 1, 0, 0, 0, -x2p * x2, -x2p * y2],
	     [0, 0, 0, x2, y2, 1, -y2p * x2, -y2p * y1],
	     [x3, y3, 1, 0, 0, 0, -x3p * x3, -x3p * y3],
	     [0, 0, 0, x3, y3, 1, -y3p * x3, -y3p * y3],
	     [x4, y4, 1, 0, 0, 0, -x4p * x4, -x4p * y4],
	     [0, 0, 0, x4, y4, 1, -y4p * x4, -y4p * y4]
	     ]

	return np.linalg.inv(x) @ p

if __name__ == '__main__':

	###### problem 1 c
	p = np.array([5,4,7,4,6,5])
	x = np.array([[0,0,1,0,0,0],[0,0,0,0,0,1],[1,0,1,0,0,0],[0,0,0,1,0,0],[0,1,1,0,0,0],[0,0,0,0,1,1]])
	a = np.linalg.inv(x)@p
	print(a)

	print(homography({(0,0):(5,4), (1,0):(7,4), (1,1):(7,5), (0,1):(6,6) }))





