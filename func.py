import time

import numpy as np
import cupy as cp
from cupy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

def dT_Eff_M(Tnorm, xyh, hd, xyzCL, dxyz, Kappa, Q, Dr, Ir, L, ML, shape, x, y, z, X, Y, Z):

    # Расчет d
    if Tnorm is not None:
        kMT0 = 0.079577471545948 * Tnorm[0]
        a0 = cp.array([
            cp.cos(cp.deg2rad(Tnorm[2])) * cp.sin(cp.deg2rad(Tnorm[1])),
            cp.cos(cp.deg2rad(Tnorm[2])) * cp.cos(cp.deg2rad(Tnorm[1])),
            cp.sin(cp.deg2rad(Tnorm[2]))
        ])
        ar = cp.array([
            cp.cos(cp.deg2rad(Ir)) * cp.sin(cp.deg2rad(Dr)),
            cp.cos(cp.deg2rad(Ir)) * cp.cos(cp.deg2rad(Dr)),
            cp.sin(cp.deg2rad(Ir))
        ])
        b = cp.array([
            a0[0] * ar[0] - a0[1] * ar[1],
            a0[0] * ar[1] + a0[1] * ar[0],
            a0[0] * ar[2] + a0[2] * ar[0],
            a0[1] * ar[2] + a0[2] * ar[1],
            a0[2] * ar[2] - a0[1] * ar[1]
        ])
        a = cp.array([
            a0[0]**2 - a0[1]**2,
            2 * a0[0] * a0[1],
            2 * a0[0] * a0[2],
            2 * a0[1] * a0[2],
            a0[2]**2 - a0[1]**2
        ])
        d = (b * Q + a) * kMT0
        d = d.astype(cp.float32)
        d = d.reshape((1, 5))
    else:
        d = cp.array([[]])

    drmin = cp.min(dxyz) * 1e-4
    drmin = cp.asarray(drmin, dtype='float32')

    n_pt = len(xyh)
    n_objs = len(xyzCL)

    xyh[:, 2] += hd

    # Основной расчет
    tic = time.perf_counter()
    dT = dTVrr(xyh, xyzCL[:, :3], dxyz, n_pt, n_objs, d, drmin, Kappa, L, ML, shape, x, y, z, X, Y, Z)
    toc = time.perf_counter()
    print('dTVrr', round(toc-tic, 4))

    return dT


def avoid_zeros(a, b, eps):
    # Функция сдвигает элементы, равные 0, на ±eps
    zero_mask_a = (a == 0)
    zero_mask_b = (b == 0)

    # Для a: если a==0, сдвигаем на -eps, если b==0, сдвигаем на +eps
    a = cp.where(zero_mask_a, a - eps, a)
    a = cp.where(zero_mask_b, a + eps, a)

    # Для b: если a==0, сдвигаем на -eps, если b==0, сдвигаем на +eps
    b = cp.where(zero_mask_a, b - eps, b)
    b = cp.where(zero_mask_b, b + eps, b)

    return a, b


def dTVrr(xyh, xyz, dxyz, n_pt, n_objs, d, eps, Kappa, L, ML, shape, x, y, z, X, Y, Z):
    # Расчет габаритных координат объектов
    d2 = dxyz / 2
    GeomD = cp.hstack((xyz - d2, xyz + d2))

    batch_objs = ML // n_objs
    batches = L // ML

    res_pt = n_pt - batch_objs * batches
    print("Bathches", batches, "Res_pt", res_pt)

    dT = cp.array([], dtype='float32')


    if batches != 0:
        for i in range(0, n_pt-batch_objs, batch_objs):
            s1 = xyh[i:i+batch_objs, None, 0] - GeomD[:, 0]
            s2 = xyh[i:i+batch_objs, None, 0] - GeomD[:, 3]
            #s1, s2 = avoid_zeros(s1, s2, eps)
            t1 = xyh[i:i+batch_objs, None, 1] - GeomD[:, 1]
            t2 = xyh[i:i+batch_objs, None, 1] - GeomD[:, 4]
            #t1, t2 = avoid_zeros(t1, t2, eps)
            u1 = xyh[i:i+batch_objs, None, 2] - GeomD[:, 5]
            u2 = xyh[i:i+batch_objs, None, 2] - GeomD[:, 2]
            #zero_mask_u1 = (u1 == 0)
            #zero_mask_u2 = (u2 == 0)
            #u1 = cp.where(zero_mask_u1, u1 + eps, u1)
            #u2 = cp.where(zero_mask_u1, u2 + eps, u2)
            #u1 = cp.where(zero_mask_u2, u1 - eps, u1)
            #u2 = cp.where(zero_mask_u2, u2 - eps, u2)
            s1 = s1.flatten(order='F')
            s2 = s2.flatten(order='F')
            t1 = t1.flatten(order='F')
            t2 = t2.flatten(order='F')
            u1 = u1.flatten(order='F')
            u2 = u2.flatten(order='F')

            s = xyh[i:i+batch_objs, None, 0] - X
            t = xyh[i:i+batch_objs, None, 1] - Y
            u = xyh[i:i+batch_objs, None, 2] - Z

            FB = calc(tic, s, t, u, s1, s2, t1, t2, u1, u2, d, eps, n_pt, n_objs, batch_objs, shape)
            dT = cp.concatenate((dT, FB))

        if res_pt != 0:
            s1 = xyh[-res_pt:, None, 0] - GeomD[:, 0]
            s2 = xyh[-res_pt:, None, 0] - GeomD[:, 3]
            #s1, s2 = avoid_zeros(s1, s2, eps)
            t1 = xyh[-res_pt:, None, 1] - GeomD[:, 1]
            t2 = xyh[-res_pt:, None, 1] - GeomD[:, 4]
            #t1, t2 = avoid_zeros(t1, t2, eps)
            u1 = xyh[-res_pt:, None, 2] - GeomD[:, 5]
            u2 = xyh[-res_pt:, None, 2] - GeomD[:, 2]
            #zero_mask_u1 = (u1 == 0)
            #zero_mask_u2 = (u2 == 0)
            #u1 = cp.where(zero_mask_u1, u1 + eps, u1)
            #u2 = cp.where(zero_mask_u1, u2 + eps, u2)
            #u1 = cp.where(zero_mask_u2, u1 - eps, u1)
            #u2 = cp.where(zero_mask_u2, u2 - eps, u2)
            s1 = s1.flatten(order='F')
            s2 = s2.flatten(order='F')
            t1 = t1.flatten(order='F')
            t2 = t2.flatten(order='F')
            u1 = u1.flatten(order='F')
            u2 = u2.flatten(order='F')

            s = xyh[-res_pt:, None, 0] - X
            t = xyh[-res_pt:, None, 1] - Y
            u = xyh[-res_pt:, None, 2] - Z

            FB = calc(tic, s, t, u, s1, s2, t1, t2, u1, u2, d, eps, n_pt, n_objs, res_pt, shape)
            dT = cp.concatenate((dT, FB))

    else:

        s1 = xyh[:, None, 0] - GeomD[:, 0]
        s2 = xyh[:, None, 0] - GeomD[:, 3]
        #s1, s2 = avoid_zeros(s1, s2, eps)
        t1 = xyh[:, None, 1] - GeomD[:, 1]
        t2 = xyh[:, None, 1] - GeomD[:, 4]
        #t1, t2 = avoid_zeros(t1, t2, eps)
        u1 = xyh[:, None, 2] - GeomD[:, 5]
        u2 = xyh[:, None, 2] - GeomD[:, 2]
        #zero_mask_u1 = (u1 == 0)
        #zero_mask_u2 = (u2 == 0)
        #u1 = cp.where(zero_mask_u1, u1 + eps, u1)
        #u2 = cp.where(zero_mask_u1, u2 + eps, u2)
        #u1 = cp.where(zero_mask_u2, u1 - eps, u1)
        #u2 = cp.where(zero_mask_u2, u2 - eps, u2)
        s1 = s1.flatten(order='F')
        s2 = s2.flatten(order='F')
        t1 = t1.flatten(order='F')
        t2 = t2.flatten(order='F')
        u1 = u1.flatten(order='F')
        u2 = u2.flatten(order='F')

        s = xyh[:, None, 0] - X
        #t = xyh[:, None, 1] - Y
        #u = xyh[:, None, 2] - Z

        x_tile = cp.tile(x, X.shape[0])
        x_resh = x_tile.reshape((X.shape[0], x.shape[0]))
        y_tile = cp.tile(y, Y.shape[0])
        y_resh = y_tile.reshape((Y.shape[0], y.shape[0]))


        s = y_resh - X

        ss = xyh[:, 0]

        dT = calc(s, t, u, s1, s2, t1, t2, u1, u2, d, eps, n_pt, n_objs, n_pt, shape)

    return dT * Kappa

def calc(s, t, u, s1, s2, t1, t2, u1, u2, d, eps, n_pt, n_objs, n_chunks, shape):

    s1q = s1 ** 2
    s2q = s2 ** 2
    t1q = t1 ** 2
    t2q = t2 ** 2
    u1q = u1 ** 2
    u2q = u2 ** 2

    tic = time.perf_counter()
    s = s.get()
    t = t.get()
    u = u.get()

    sq = s ** 2
    tq = t ** 2
    uq = u ** 2

    sq = cp.asarray(sq)
    tq = cp.asarray(tq)
    uq = cp.asarray(uq)
    toc = time.perf_counter()
    print('opt', toc - tic)


    #--------------------------------------------------------------------------------------------------------------------------------------
    st_s = sq.strides
    st_t = tq.strides
    st_u = uq.strides

    sqq = as_strided(sq, shape=(n_chunks, shape[0]+1, (shape[1]+1) * (shape[2]+1)), strides=(st_s[0], 4, 0)).flatten()
    tqq_ = as_strided(tq, shape=(n_chunks, shape[1]+1, shape[2]+1), strides=(st_t[0], 4, 0)).flatten()
    tqq = as_strided(tqq_, shape=(n_chunks, shape[0]+1, (shape[1]+1) * (shape[2]+1)), strides=(st_t[0]*(shape[2]+1), 0, 4)).flatten()
    uqq = as_strided(uq, shape=(n_chunks, (shape[0]+1) * (shape[1]+1), shape[2]+1), strides=(st_u[0], 0, 4)).flatten()

    r = cp.sqrt(sqq + tqq + uqq)

    shape_vertex = (shape[0] + 1) * (shape[1] + 1) * (shape[2] + 1)
    num_nodes = cp.arange(0, shape_vertex * n_chunks, dtype='int32')
    num_nodes_2 = num_nodes.reshape(n_chunks, shape_vertex)
    num_nodes_2 = num_nodes_2.T.ravel()

    offset_x = (shape[1] + 1) * (shape[2] + 1)
    offset_y = shape[2] + 1

    _111 = as_strided(num_nodes_2, shape=(shape[0], shape[1], shape[2]*n_chunks), strides=(n_chunks*offset_x*4, n_chunks*offset_y*4, 4)).flatten()
    # --------------------------------------------------------------------------------------------------------------------------------------

    _211 = _111 + offset_x
    _121 = _111 + offset_y
    _221 = _121 + offset_x
    _112 = _111 + 1
    _212 = _211 + 1
    _122 = _121 + 1
    _222 = _221 + 1

    r111 = r[_111]
    r211 = r[_211]
    r121 = r[_121]
    r221 = r[_221]
    r112 = r[_112]
    r212 = r[_212]
    r122 = r[_122]
    r222 = r[_222]


    #r111 = cp.sqrt(s1q + t1q + u1q)
    #r211 = cp.sqrt(s2q + t1q + u1q)
    #r121 = cp.sqrt(s1q + t2q + u1q)
    #r221 = cp.sqrt(s2q + t2q + u1q)
    #r112 = cp.sqrt(s1q + t1q + u2q)
    #r212 = cp.sqrt(s2q + t1q + u2q)
    #r122 = cp.sqrt(s1q + t2q + u2q)
    #r222 = cp.sqrt(s2q + t2q + u2q)





    A1 = arctg(s1, s2, t1, t2, u1, r121, r211)
    A2= -arctg(s1, s2, t2, t1, u1, r111, r221)
    A3 = arctg(s1, s2, t2, t1, u2, r112, r222)
    A4= -arctg(s1, s2, t1, t2, u2, r122, r212)
    B1 = arctg(t2, t1, s2, s1, u1, r121, r211)
    B2= -arctg(t1, t2, s2, s1, u1, r111, r221)
    B3 = arctg(t1, t2, s2, s1, u2, r112, r222)
    B4= -arctg(t2, t1, s2, s1, u2, r122, r212)

    Vxx = -(A1 + A2 + A3 + A4)
    Vzz = B1 + B2 + B3 + B4 - Vxx

    r1 = cp.stack([r111 + u1, r221 + u1, r122 + u2, r212 + u2])
    r2 = cp.stack([r112 + u2, r222 + u2, r121 + u1, r211 + u1])
    a = cp.prod(r1, axis=0)
    b = cp.prod(r2, axis=0)
    Vxy = cp.log(a / b)
    r1 = cp.stack([r111 + t1, r212 + t1, r122 + t2, r221 + t2])
    r2 = cp.stack([r121 + t2, r222 + t2, r112 + t1, r211 + t1])
    a = cp.prod(r1, axis=0)
    b = cp.prod(r2, axis=0)
    Vxz = cp.log(a / b)
    r1 = cp.stack([r111 + s1, r122 + s1, r212 + s2, r221 + s2])
    r2 = cp.stack([r211 + s2, r222 + s2, r112 + s1, r121 + s1])
    a = cp.prod(r1, axis=0)
    b = cp.prod(r2, axis=0)
    Vyz = cp.log(a / b)
    V = cp.stack([Vxx, Vxy, Vxz, Vyz, Vzz])

    FB = d @ V

    FB = FB[0].reshape((n_objs, n_chunks))
    dT = cp.sum(FB, axis=0)

    return dT


def arctg(s1, s2, t1, t2, u, r1, r2):
    a = cp.arctan((t2 * u) / (r1 * s1))
    b = cp.arctan((t1 * u) / (r2 * s2))
    return a + b


def Plot_Isolines_T(xyT, shape):
    x = xyT[:, 0]
    y = xyT[:, 1]
    z = xyT[:, 2]

    # Создаем сетку
    xi = cp.unique(x)
    yi = cp.unique(y)

    X, Y = cp.meshgrid(xi, yi)
    Z = cp.reshape(z, shape)

    X, Y, Z = X.get(), Y.get(), Z.get()

    # Построение графика
    plt.contourf(X, Y, Z)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour plot')
    plt.colorbar()
    plt.show()