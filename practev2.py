import cv2

# # task 1

image = cv2.imread("data/lesson2/marbles.png")
cv2.imshow("marbles", image)
cv2.waitKey(0)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower =(100,100,100)
upper= (130,255,255)

mask_blue = cv2.inRange(hsv,lower, upper)

cv2.imshow('mask', mask_blue)
cv2.waitKey(0)


lower = (0,100,150)
upper = (7,255, 255)

mask_red = cv2.inRange(hsv, lower, upper)

cv2.imshow('mask', mask_red)
cv2.waitKey(0)

lower = (40,90,100)
upper = (85,255,255)

mask_green = cv2.inRange(hsv, lower, upper)

cv2.imshow('mask', mask_green)
cv2.waitKey(0)

mask_both = cv2.bitwise_or(mask_red, mask_green)
cv2.imshow('mask', mask_both)
cv2.waitKey(0)

lower = (0,0,0)
upper = (100,100,40)

mask_black = cv2.inRange(hsv, lower, upper)

cv2.imshow('mask', mask_black)
cv2.waitKey(0)

lower = (0,0,200)
upper = (180,30,255)

mask_white = cv2.inRange(hsv, lower, upper)

cv2.imshow('mask', mask_white)
cv2.waitKey(0)

# # task 2

img = cv2.imread("data/lesson2/cell.png")

img = cv2.resize(img, (500,500))

cv2.imshow("image", img)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

new_l = cv2.equalizeHist(l)

new_lab = cv2.merge((new_l, a, b))

new_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

cv2.imshow("new image", new_img)

cv2.waitKey(0)


