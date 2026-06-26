import numpy as np
from sqlalchemy.util import counter

# створення масиву

# nums = np.array([1, 2, 3, 4, 5])
# print(nums)
# print(type(nums))
#
# print(nums.shape)  # розмір масиву
# print(nums.dtype)  # тип даних одного елемента в комірці
# # int64 -- 64 біта на одну комірку
#
# # вказати тип даних
# nums = np.array([1, 2, 3, 4, 5], dtype=np.float32)
# print(nums)
# print(nums.shape)
# print(nums.dtype)

# двовимірний масив(таблиця/матриця)

# nums = np.array(
#     [[1, 2, 3, 4],
#      [5, 6, 7, 8],
#      [9, 10, 11, 12]]
# )
#
# print(nums)
# print(nums.shape)  # розмір масиву
# print(nums.dtype)  # тип даних одного елемента в комірці



# # порівнння швидкості для масивів та списків
# import time
#
#
# N = 10_000_000
# nums_list = list(range(N))
# nums_array = np.array(range(N))
#
# start = time.time()
# res = sum(nums_list)
# end = time.time()
#
# print(f"Python list: {end - start:0.5f} sec")
#
# start = time.time()
# res = np.sum(nums_array)
# end = time.time()
#
# print(f"Numpy array: {end - start:0.5f} sec")
#
# start = time.time()
# total = 0
# for num in nums_array:
#     total += num
# end = time.time()
#
# print(f"Numpy array with for: {end - start:0.5f} sec")


# цикл for -- то є зло
# користуєтеся numpy функція


# створення та розміри

# # зі списку
# nums_list = [1, 2, 3, 4]
# nums = np.array(nums_list)
#
# # аналог range
# nums = np.arange(10, 20)
# print(nums)

# масив нулів/одиниць/випадкових чисел
# nums = np.zeros((6,))  # вказуєте розмір
# print(nums)
#
# nums = np.ones((5, 3))
# print(nums)
#
# nums = np.random.rand(2, 3)  # випадкові від 0 до 1
# print(nums)

# # зміна розмірів та типів
#
# nums = np.arange(12)
# print(nums)
# print(nums.shape)
# print(nums.dtype)
#
# new_nums = nums.reshape((3, 4))
# print(new_nums)
# print(new_nums.shape)
# print(new_nums.dtype)
#
# nums_float16 = nums.astype(np.float16)
# print(nums_float16)
# print(nums_float16.shape)
# print(nums_float16.dtype)


# # перенаповнення
# # int8 -- -128..127
# nums = np.array([10, 20, 30, 120], dtype=np.int8)
# print(nums)
#
# # збільшити всі числа на 10
# nums = nums + 10
# print(nums)


# індексація
# nums = np.array([10, 20, 30, 40, 50])
#
# print(nums[0])  # перший елемент
# print(nums[3])  # 40
# print(nums[-1])  # останній
# print(nums[-3])  # третій з кінця
# print(nums[1:4])  # 20 - 40


# # індексація таблиць
# nums = np.arange(12).reshape((4, 3))
# print(nums)

# спочатку рядки потім стовпчики
# nums[індекс рядка, індекс стовпчика]

# print(nums[1, 2])  # 5(рядок 1, стовпчик 2)
# print(nums[3])     # весь рядок 3
# print(nums[0:2])   # перші 2 рядка
# print(nums[:, 1])     # стовпчик 1
# print(nums[:, 1:3])    # два останніх стовпчика
# print(nums[1:3, 0:2])



# базові операці

# nums1 = np.array([1, 2, 3])
# nums2 = np.array([2, 2, 2])
#
# print(nums1 + 10)
# print(nums1 * nums2)


# булеві маски

# nums = np.array([15, 8, 17, 18, 1, 2, 3])
#
# mask = nums > 10
#
# # отримати елементи які відповідають масці
# print(nums[mask])
#
# # всі елементи які відповідають масці збільшити в 2 рази
# nums[mask] *= 2
# print(nums)
#
# #всі елементи які відповідають масці замінити на -1
# nums[mask] = -1
# print(nums)

nums = np.arange(1, 11)
print(nums)
print(nums.shape)
print(nums.dtype)

new_nums = nums.reshape(2,5)
print(new_nums)
print(new_nums.shape)
print(new_nums.dtype)

nums = np.arange(1, 13)
nums2 = nums.reshape(3,4)

print(nums2[1,2])

print(nums2[1])

print(nums2[:,3])

print(nums2[:,2:4])

print(nums2[1:3,1:3])

# nums2[1:3,1:3] = -1
# print(nums2)
# nums2[:,0] = nums2[:,1]
# print(nums2)

mask = nums2 > 6
mask2 = ~mask
print(np.sum(mask))

print(nums2[mask])

nums2[mask] += 10
print(nums2)

nums2[mask2] *= -1
print(nums2)

new_nums2  = np.array([
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0]
])

nums2[mask2] = new_nums2[mask2]

print(nums2)


