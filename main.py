import time

import numpy as np
import cupy as cp

from func import dT_Eff_M, Plot_Isolines_T


# Исходные параметры
tic = time.perf_counter()
x1, x2, dx = -1000, 1000, 200
y1, y2, dy = -1000, 1000, 200
H = 0
hd = 0

# Создаем сетку
X, Y = cp.meshgrid(cp.arange(x1, x2 + dx, dx), cp.arange(y1, y2 + dy, dy))

# Преобразуем в массив точек (Nx2)
xyHh = cp.column_stack((X.ravel(), Y.ravel()))
#xyHh = cp.array([[-1000, -950]])
xyHh = cp.array([[-1000, -950], [-900, -850]])

# Добавляем столбцы H и H + hd
num_points = xyHh.shape[0]
H_column = cp.full((num_points, 1), H)
Hd_column = cp.full((num_points, 1), H + hd)

# Объединяем все вместе
xyHh = cp.hstack((xyHh, H_column, Hd_column), dtype='float32')

toc = time.perf_counter()
print('start', round(toc-tic, 4))
print()

# Исходные координаты кубов
x_start = -500
y_start = -400
z_start = -700
# Размеры кубика по каждой оси
dx_obj = 200  # длина по X
dy_obj = 200 # длина по Y
dz_obj = 200  # длина по Z
# Размерность общего куба объектов
shape_cube = (2, 2, 2)
num_cubes = shape_cube[0] * shape_cube[1] * shape_cube[2]
len_calc = num_cubes * num_points
max_len_calc = 200_000_000

obj = cp.array([x_start, x_start+dx_obj, y_start, y_start+dy_obj, z_start, z_start+dz_obj], dtype='float32')  # xa, xb, ya, yb, za, zb

# Создаем сетку индексов
ix, iy, iz = cp.meshgrid(cp.arange(shape_cube[0]), cp.arange(shape_cube[1]), cp.arange(shape_cube[2]), indexing='ij')

# Выравниваем в вектор
ix = ix.ravel()
iy = iy.ravel()
iz = iz.ravel()

# Смещения по осям
offset_x = ix * dx_obj
offset_y = iy * dy_obj
offset_z = iz * dz_obj

# Повторяем исходный кубик (по количеству кубиков)
objs = cp.tile(obj, (num_cubes, 1))

# Добавляем смещения к каждому кубику
objs[:, 0] += offset_x  # xmin
objs[:, 1] += offset_x  # xmax
objs[:, 2] += offset_y  # ymin
objs[:, 3] += offset_y  # ymax
objs[:, 4] += offset_z  # zmin
objs[:, 5] += offset_z  # zmax

# Размеры по осям
dxyz = cp.array([objs[0, 1] - objs[0, 0], objs[0, 3] - objs[0, 2], objs[0, 4] - objs[0, 5]], dtype='float32')

# Центр модели
xyzMod = cp.array([
    (objs[:, 0] + objs[:, 1]) / 2,
    (objs[:, 2] + objs[:, 3]) / 2,
    (objs[:, 4] + objs[:, 5]) / 2,
    (objs[:, 4] + objs[:, 5]) / 2
], dtype='float32')

xyzMod = xyzMod.T

Tnorm = cp.array([50000, -10, 60], dtype='float32')

# Параметры
Kappa = 0.01
Q = 2
Dr = 0
Ir = -90

tic = time.perf_counter()
dT = dT_Eff_M(Tnorm, xyHh[:, :3], xyHh[:, 3], xyzMod, dxyz, Kappa, Q, Dr, Ir, len_calc, max_len_calc, shape_cube)
toc = time.perf_counter()
print('dT_Eff', round(toc-tic, 4))

xyHdT = np.column_stack((xyHh, dT))
xyT = np.column_stack((xyHdT[:, :2], dT))

shape = (((x2 - x1) // dx) + 1, ((y2 - y1) // dy) + 1)

Plot_Isolines_T(xyT, shape)