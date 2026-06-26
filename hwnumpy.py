import numpy as np

# task 1

# Створення масиву
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# число 14
print(arr[3, 1])

# третій рядок
print(arr[2, :])

# перший стовпчик
print(arr[:, 0])

# верхню половину
print(arr[0:2, :])

# замініть числа в рядках 2-3 на 100
arr[2:4, :] = 100
print(arr)

# зробіть другий рядок таким як останній рядок
arr[1, :] = arr[-1, :]
print(arr)

# task 2

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# маска для парних чисел
mask = arr % 2 == 0

# виведіть самі числа
print(arr[mask])

# замініть їх на 100
arr[mask] = 100
print(arr)

# task 3

m1 = np.array([128, 200, 10], dtype=np.uint8)
m2 = np.array([250, 10, 34], dtype=np.uint8)

# Об'єднання у пропорції 20% першого + 80% другого
res = m1.astype(np.float64) * 0.2 + m2.astype(np.float64) * 0.8
res = np.clip(res, 0, 255).astype(np.uint8)
print(res)