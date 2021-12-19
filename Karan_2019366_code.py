import cv2
import numpy as np

# This function returns an output image given an input image and a transformation matrix


def bilinearInterpolation(img, transMatrix):
    inverseMatrix = np.linalg.inv(transMatrix)
    out_img = np.zeros([500, 500])
    xLim = len(img) - 1
    yLim = len(img[0]) - 1
    origin = [250, 250]
    for x in range(500):
        for y in range(500):
            out_cord = np.array([x - origin[0], y - origin[1], 1])
            in_cord = np.dot(out_cord, inverseMatrix)

            xnot = in_cord[0]
            ynot = in_cord[1]

            """
			#Q5
			xnot = in_cord[0] + 250
			ynot = in_cord[1] + 250
			"""
            # coordinates are outside the image
            if np.floor(xnot) > xLim or np.floor(xnot) < 0 or np.floor(ynot) > yLim or np.floor(ynot) < 0:
                continue

            # coordinates are on the border(mirroring)
            elif np.floor(xnot) == xLim and np.floor(ynot) == yLim:
                out_img[x][y] = img[int(np.floor(xnot))][int(np.floor(ynot))]
            elif np.floor(xnot) == xLim:
                if ynot - np.floor(ynot) <= ynot:
                    out_img[x][y] = img[int(
                        np.floor(xnot))][int(np.floor(ynot))]
                else:
                    out_img[x][y] = img[int(
                        np.floor(xnot))][int(np.ceil(ynot))]
            elif np.floor(ynot) == yLim:
                if xnot - np.floor(xnot) <= xnot:
                    out_img[x][y] = img[int(
                        np.floor(xnot))][int(np.floor(ynot))]
                else:
                    out_img[x][y] = img[int(
                        np.ceil(xnot))][int(np.floor(ynot))]

            # coordinates are inside the image
            elif xnot == np.floor(xnot) and ynot == np.floor(ynot):
                xnot = int(xnot)
                ynot = int(ynot)
                out_img[x][y] = img[xnot][ynot]
            else:
                if(np.ceil(xnot) != xnot):
                    x1 = np.floor(xnot)
                    x2 = np.ceil(xnot)
                else:
                    if xnot == 250:
                        x1 = 250
                        x2 = 251
                    else:
                        x1 = xnot - 1
                        x2 = xnot

                if(np.ceil(ynot) != ynot):
                    y1 = np.floor(ynot)
                    y2 = np.ceil(ynot)
                else:
                    if ynot == 250:
                        y1 = 250
                        y2 = 251
                    else:
                        y1 = ynot - 1
                        y2 = ynot
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                X = [[x1, y1, x1 * y1, 1], [x1, y2, x1 * y2, 1],
                     [x2, y2, x2 * y2, 1], [x2, y1, x2 * y1, 1]]
                Y = [img[x1][y1], img[x1][y2], img[x2][y2], img[x2][y1]]
                A = np.dot(np.linalg.inv(X), Y)
                out_img[x][y] = np.dot([xnot, ynot, xnot * ynot, 1], A)
    return out_img


# Question 3
img = cv2.imread("./assign1.jpg", 0)
transMatrix = np.identity(3)
print("Enter Interpolation Factor:")
c = float(input())
transMatrix[0][0] = c
transMatrix[1][1] = c

out_img = bilinearInterpolation(img, transMatrix)


cv2.imshow("input", img)
cv2.imshow("output", out_img / 255)

"""
#Question 4
img = cv2.imread("./IMG_Q4.jpg",0)

#Transformation Matrix: 

def translate(transMatrix, moveX, moveY):
	multMatrix = np.identity(3)
	multMatrix[2][0] += moveX
	multMatrix[2][1] += moveY
	transMatrix = np.dot(transMatrix,multMatrix)
	return transMatrix

def scale(transMatrix, scaleX, scaleY):
	multMatrix = np.identity(3)
	multMatrix[0][0] = scaleX
	multMatrix[1][1] = scaleY
	transMatrix = np.dot(transMatrix,multMatrix)
	return transMatrix

def rotate(transMatrix, theta):
	multMatrix = np.identity(3)
	rad = np.radians(theta)
	multMatrix[0][0] = np.cos(rad)
	multMatrix[0][1] = -np.sin(rad)
	multMatrix[1][1] = np.cos(rad)
	multMatrix[1][0] = np.sin(rad)
	transMatrix = np.dot(transMatrix,multMatrix)
	return transMatrix


#This function takes the different types of transformations in a specific order 
#and returns one final transformation matrix
def createTransMat():
	transMatrix = np.identity(3)
	while True: 
		print('Choose Transformation:\n1. Rotate\n2. Scale\n3. Translate\n4. Exit\n')
		ch = int(input())
		if ch == 1:
			print('Input angle in deg:')
			theta = float(input())
			transMatrix = rotate(transMatrix, theta)
		elif ch == 2:
			print('Scale x by:', end = ' ')
			scaleX = float(input())
			print('Scale y by:', end = ' ')
			scaleY = float(input())
			transMatrix = scale(transMatrix, scaleX, scaleY)
			
		elif ch == 3:
			print('Move x by:', end = ' ')
			moveX = float(input())
			print('Move y by:', end = ' ')
			moveY = float(input())
			transMatrix = translate(transMatrix, moveX, moveY)

		else:
			break
	return transMatrix

transMatrix = createTransMat()

out_img = bilinearInterpolation(img,transMatrix)


cv2.imshow("input",img)
cv2.imshow("output",out_img/250)
cv2.imwrite("output.jpg",out_img)
"""
"""
#Question 5

y1,x1 = 280,280 
w1,v1 = 0,0
y2,x2 = 190,371 
w2,v2 = 0,63
y3,x3 = 280,461 
w3,v3 = 63,63

X = [[x1-250,y1-250,1],[x2-250,y2-250,1],[x3-250,y3-250,1]]
V = [[v1,w1,1],[v2,w2,1],[v3,w3,1]]

out_img = cv2.imread("./output.jpg",0)

img = cv2.imread("./IMG_Q4.jpg",0)
transMatrix = np.dot(np.linalg.inv(V),X)
print(transMatrix)

newIn_img = bilinearInterpolation(out_img,np.linalg.inv(transMatrix))
cv2.imshow("input_reference",img)
cv2.imshow("input_referred",newIn_img/255)
cv2.imwrite("input_referred.jpg",newIn_img)
"""
cv2.waitKey(0)
cv2.destroyAllWindows()
