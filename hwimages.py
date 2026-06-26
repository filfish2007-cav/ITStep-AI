import cv2
#
# # task 1
#
# # Відкрийте зображення та прочитайте маски
# img = cv2.imread("data/Lenna.png")
# mask1 = cv2.imread("data/mask1.png", cv2.IMREAD_GRAYSCALE)
# mask2 = cv2.imread("data/mask2.png", cv2.IMREAD_GRAYSCALE)
#
# # Об'єднайте дві маски в одну за допомогою cv2.bitwise_or() та виведіть результат
# combined_mask = cv2.bitwise_or(mask1, mask2)
# cv2.imshow("Combined Mask", combined_mask)
# cv2.waitKey(0)
#
# # Зміна типу даних у масках на bool для подальшого використання
# m1_bool = mask1.astype(bool)
# m2_bool = mask2.astype(bool)
# m1_m2_bool = combined_mask.astype(bool)
#
# # Виведіть частину зображення, яка відповідає mask1 (усі інші пікселі замініть на 0)
# res1 = img.copy()
# res1[~m1_bool] = 0
# cv2.imshow("Mask1 Result", res1)
# cv2.waitKey(0)
#
# # Виведіть частину зображення, яка відповідає mask2 (усі інші пікселі замініть на 0)
# res2 = img.copy()
# res2[~m2_bool] = 0
# cv2.imshow("Mask2 Result", res2)
# cv2.waitKey(0)
#
# # Виведіть частину зображення, яка відповідає mask1 i mask2 (усі інші пікселі замініть на 0)
# res3 = img.copy()
# res3[~m1_m2_bool] = 0
# cv2.imshow("Mask1 and Mask2 Result", res3)
# cv2.waitKey(0)

# task 2

img = cv2.imread("data/lesson1/baboo.jpg")

eyes = img[12:45, 43:195]

cv2.imshow("Eyes", eyes)
cv2.waitKey(0)
cv2.destroyAllWindows()