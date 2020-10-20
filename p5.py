import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import collections
from numpy.random import default_rng
import copy


# from https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
def pca(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  # Project X onto PC space
  X_pca = np.dot(X, eigen_vecs)
  return X_pca

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


def ransac(numIter,Mat,xyprime,matches,keepPercent,sampSize):

	mseHdict = {}
	matches = sorted(matches, key=lambda x: x.distance)
	matches = matches[:int(len(matches)*(keepPercent/100))]

	for iter in range(numIter):

		idx = []
		rng = default_rng()
		idx1 = rng.choice(int(len(Mat)/2), size=sampSize, replace=False)
		idx1 = [2 * i for i in idx1]
		idx2 = [i + 1 for i in idx1]
		idx.append(idx1)
		idx.append(idx2)
		idx = np.array(idx).flatten()
		idx = sorted(idx)

		M = Mat[idx, :]
		xypVect = xyprime[idx]

		a = np.linalg.pinv(M) @ np.array(xypVect)
		H = np.array([[a[0], a[1], a[2]],
		              [a[3], a[4], a[5]],
		              [a[6], a[7], 1]])

		error = []
		for match in matches:
			pt = kp1[match.queryIdx].pt
			x = pt[0]
			y = pt[1]

			ptp = kp2[match.trainIdx].pt
			xp = ptp[0]
			yp = ptp[1]

			ptpPredicted = H @ np.array([x, y, 1])
			xpPredicted = ptpPredicted[0] / ptpPredicted[2]
			ypPredicted = ptpPredicted[1] / ptpPredicted[2]
			error.append(xp - xpPredicted)
			error.append(yp - ypPredicted)
		error = np.array(error)
		mse = error @ error / len(error)
		mseHdict[mse] = H


	best = min(mseHdict.keys())
	#print(collections.OrderedDict(sorted(mseHdict.items())))
	print('min:' + str(best))
	print('max:' + str(max(mseHdict.keys())))
	print('avg' + str(np.mean(list(mseHdict.keys()))))
	return mseHdict[best]


def getIDX(Hfinal,r,c):
	ptp = Hfinal @ [r, c, 1]
	rp = ptp[0] / ptp[2]
	cp = ptp[1] / ptp[2]
	return (int(rp) ,int(cp))

#https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

if __name__ == '__main__':
	# wall1 = cv2.imread('wall1.png',0)
	# wall2 = cv2.imread('wall2.png',0)

	wall1 = cv2.imread('1.jpg', 0)
	wall2 = cv2.imread('2.jpg', 0)
	wall1 = ResizeWithAspectRatio(wall1, width=720)
	wall2 = ResizeWithAspectRatio(wall2, width=720)

	# inspired by https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
	orb = cv2.ORB_create(nlevels=40,nfeatures=10000)

	kp1 = orb.detect(wall1, None)
	kp1, des1 = orb.compute(wall1, kp1)

	kp2 = orb.detect(wall2, None)
	kp2, des2 = orb.compute(wall2, kp2)

	# draw only keypoints location,not size and orientation
	wall1kp = None
	wall1kp = cv2.drawKeypoints(wall1, kp1, wall1kp, color=(0, 255, 0), flags=0)
	wall2kp = cv2.drawKeypoints(wall2, kp2, wall2, color=(0, 255, 0), flags=0)
	cv2.imshow('wall pic 1 with key points', wall1kp)
	cv2.imshow('wall pic 2 with key points', wall2kp)
	cv2.waitKey(0)

	#taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	matchImg = cv2.drawMatches(wall1, kp1, wall2, kp2, matches[:10],np.array([[]]), flags=2)
	#plt.imshow(matchImg),plt.show()

	cv2.imshow('matched key points', matchImg)
	cv2.waitKey(0)


	for match in matches[:10]:
		print(str(kp1[match.queryIdx].pt) + " : " + str(kp2[match.trainIdx].pt))


	pdDict = {}
	M = []
	xypVect = []
	df = pd.DataFrame()
	for match in matches:
		xy = kp1[match.queryIdx].pt
		xyp = kp2[match.trainIdx].pt

		x = xy[0]
		y = xy[1]
		xp = xyp[0]
		yp = xyp[1]

		pdDict['x'] = [x]
		pdDict['y'] = [y]
		pdDict['xp'] = [xp]
		pdDict['yp'] = [yp]

		xypVect.append(xp)
		xypVect.append(yp)
		M.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y])
		M.append([0, 0, 0, x, y, 1, -yp * x, -yp * y])

		#print(str(kp1[match.queryIdx].pt) + " : " + str(kp2[match.trainIdx].pt))

		df = df.append(pd.DataFrame.from_dict(pdDict))

	M = np.array(M)
	xypVect = np.array(xypVect)

	numIter = 15000
	sampSize = 4
	topPercent = 5
	Hfinal = ransac(numIter,M,xypVect,matches,topPercent,sampSize)
	print()

	for match in matches[:10]:
		pt = kp1[match.queryIdx].pt
		ptp = Hfinal@[pt[0],pt[1],1]
		xp = ptp[0]/ptp[2]
		yp = ptp[1]/ptp[2]

		print(str(pt) + " : " + str(xp) + " , " + str(yp))


	shp = wall1.shape
	wall1towall2pointMap = {}
	wall2towall1pointMap = {}

	maxY = -99999999
	minY = 9999999999
	maxX = -9999999999
	minX = 99999999999
	for r in range(shp[0]*2):
		r = r/2
		for c in range(shp[1]*2):
			c= c/2
			X , Y = getIDX(Hfinal, c, r)
			wall1towall2pointMap[(c,r)]= (X, Y)
			wall2towall1pointMap[(X,Y)] = (int(c),int(r))
			if maxY < Y:
				maxY = Y
			if maxX < X:
				maxX = X
			if minY > Y:
				minY = Y
			if minX > X:
				minX = X


	maxY = max(maxY, wall2.shape[0])
	minY = min(minY, wall2.shape[0])
	maxX = max(maxX, wall2.shape[1])
	minX = min(minX, wall2.shape[1])
	print((maxY, minY))
	print((maxX, minX))

	shiftrow = 0
	if minY < 0:
		shiftrow = abs(minY)

	shiftcol = 0
	if minX < 0:
		shiftcol = abs(minX)

	dtype = wall2.dtype
	out = np.zeros((maxY + shiftrow, maxX + shiftcol))
	#out = np.zeros((maxc+shiftcol,maxr+shiftrow))
	out[shiftrow:wall2.shape[0]+shiftrow,shiftcol:wall2.shape[1]+shiftcol] = wall2 #just copy in values from wall 2 to start with

	last = (0-shiftcol,0-shiftrow)
	for y in range(out.shape[0]):
		for x in range(out.shape[1]):
			#rcp = None
			if (x-shiftcol,y-shiftrow) in wall2towall1pointMap:
				# if rcp in wall1towall2pointMap:
				# 	rcp = wall1towall2pointMap[(x,y)]
				# else:
				# 	rcp = last
				# last = rcp
				rcp = wall2towall1pointMap[(x-shiftcol,y-shiftrow)]
				Xp = rcp[0] #+ shiftcol
				Yp = rcp[1] # + shiftrow
				if out[y,x] != 0:
					out[y,x] = np.mean([out[y,x], wall1[Yp, Xp]])
					#pass
				else:
					out[y,x] = wall1[Yp,Xp]

	# for point in wall1towall2pointMap:
	# 	x = point[0]
	# 	y = point[1]
	#
	# 	rcp = wall1towall2pointMap[(x, y)]
	# 	Xp = rcp[0] + shiftcol
	# 	Yp = rcp[1] + shiftrow
	#
	# 	if out[Yp, Xp] !=0 :
	# 		out[Yp, Xp] = np.mean([out[Yp, Xp], wall1[y, x]])
	# 	else:
	# 		out[Yp, Xp] = wall1[y, x]

	out = out.astype(dtype)

	kernal = np.array([[1,2,1],[2,4,2],[1,2,1]])
	kernal = kernal/np.sum(kernal)

	dst = cv2.filter2D(out, -1, kernal)
	#dst = dst.astype()

	cv2.imshow('mosiac', out)
	cv2.waitKey(0)

#need to run through a bunch of samples compute the H matrix params this maps
	#pairings (x,y):(x',y') to [a,...] this gives a feature space the point mapings and
	# a y the [a..] params want to find 4 points that are representitave of

	#pcs = pca(X)




