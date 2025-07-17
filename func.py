import time

import cupy as cp
from cupy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

def dT_Eff_M(Tnorm, xyh, hd, xyzCL, dxyz, Kappa, Q, Dr, Ir, L, ML, shape):

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
    dT = dTVrr(xyh, xyzCL[:, :3], dxyz, n_pt, n_objs, d, drmin, Kappa, L, ML, shape)
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


def dTVrr(xyh, xyz, dxyz, n_pt, n_objs, d, eps, Kappa, L, ML, shape):
    # Расчет габаритных координат объектов
    d2 = dxyz / 2
    GeomD = cp.hstack((xyz - d2, xyz + d2))

    x_ = -500
    y_ = -400
    z_ = -700

    dx, dy, dz = 200, 200, 200

    X = cp.arange(x_, x_ + dx*(shape[0]+1), dx, dtype='float32')
    Y = cp.arange(y_, y_ + dy*(shape[1]+1), dy, dtype='float32')
    Z = cp.arange(z_, z_ + dz*(shape[2]+1), dz, dtype='float32')



    batch_objs = ML // n_objs
    batches = L // ML
    print("Bathches", batches)

    res_pt = n_pt - batch_objs * batches

    dT = cp.array([], dtype='float32')

    if batches != 0:
        for i in range(0, n_pt-batch_objs, batch_objs):
            s1 = xyh[i:i+batch_objs, None, 0] - GeomD[:, 0]
            s2 = xyh[i:i+batch_objs, None, 0] - GeomD[:, 3]
            s1, s2 = avoid_zeros(s1, s2, eps)
            t1 = xyh[i:i+batch_objs, None, 1] - GeomD[:, 1]
            t2 = xyh[i:i+batch_objs, None, 1] - GeomD[:, 4]
            t1, t2 = avoid_zeros(t1, t2, eps)
            u1 = xyh[i:i+batch_objs, None, 2] - GeomD[:, 5]
            u2 = xyh[i:i+batch_objs, None, 2] - GeomD[:, 2]
            zero_mask_u1 = (u1 == 0)
            zero_mask_u2 = (u2 == 0)
            u1 = cp.where(zero_mask_u1, u1 + eps, u1)
            u2 = cp.where(zero_mask_u1, u2 + eps, u2)
            u1 = cp.where(zero_mask_u2, u1 - eps, u1)
            u2 = cp.where(zero_mask_u2, u2 - eps, u2)
            s1 = s1.flatten(order='F')
            s2 = s2.flatten(order='F')
            t1 = t1.flatten(order='F')
            t2 = t2.flatten(order='F')
            u1 = u1.flatten(order='F')
            u2 = u2.flatten(order='F')

            FB = calc(s1, s2, t1, t2, u1, u2, d, eps, n_objs, batch_objs)
            dT = cp.concatenate((dT, FB))

        if res_pt != 0:
            s1 = xyh[-res_pt:, None, 0] - GeomD[:, 0]
            s2 = xyh[-res_pt:, None, 0] - GeomD[:, 3]
            s1, s2 = avoid_zeros(s1, s2, eps)
            t1 = xyh[-res_pt:, None, 1] - GeomD[:, 1]
            t2 = xyh[-res_pt:, None, 1] - GeomD[:, 4]
            t1, t2 = avoid_zeros(t1, t2, eps)
            u1 = xyh[-res_pt:, None, 2] - GeomD[:, 5]
            u2 = xyh[-res_pt:, None, 2] - GeomD[:, 2]
            zero_mask_u1 = (u1 == 0)
            zero_mask_u2 = (u2 == 0)
            u1 = cp.where(zero_mask_u1, u1 + eps, u1)
            u2 = cp.where(zero_mask_u1, u2 + eps, u2)
            u1 = cp.where(zero_mask_u2, u1 - eps, u1)
            u2 = cp.where(zero_mask_u2, u2 - eps, u2)
            s1 = s1.flatten(order='F')
            s2 = s2.flatten(order='F')
            t1 = t1.flatten(order='F')
            t2 = t2.flatten(order='F')
            u1 = u1.flatten(order='F')
            u2 = u2.flatten(order='F')

            FB = calc(s1, s2, t1, t2, u1, u2, d, eps, n_objs, res_pt)
            dT = cp.concatenate((dT, FB))

    else:
        tic = time.perf_counter()
        s1 = xyh[:, None, 0] - GeomD[:, 0]
        s2 = xyh[:, None, 0] - GeomD[:, 3]
        s = xyh[:, None, 0] - X
        #s = avoid_zeros(s, eps)
        #s1, s2 = avoid_zeros(s1, s2, eps)
        t1 = xyh[:, None, 1] - GeomD[:, 1]
        t2 = xyh[:, None, 1] - GeomD[:, 4]
        t = xyh[:, None, 1] - Y
        #t = avoid_zeros(t, eps)
        #t1, t2 = avoid_zeros(t1, t2, eps)
        u1 = xyh[:, None, 2] - GeomD[:, 5]
        u2 = xyh[:, None, 2] - GeomD[:, 2]
        u = xyh[:, None, 2] - Z
        #zero_mask_u = (u == 0)
        #u = cp.where(zero_mask_u, u + eps, u)
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
        #s, t, u = s.flatten(order='F'), t.flatten(order='F'), u.flatten(order='F')
        #toc = time.perf_counter()
        #print('GeomD', round(toc - tic, 5))

        dT = calc(tic, s, t, u, s1, s2, t1, t2, u1, u2, d, eps, n_pt, n_objs, n_pt, shape)

    return dT * Kappa

def calc(tic, s, t, u, s1, s2, t1, t2, u1, u2, d, eps, n_pt, n_objs, batches, shape):

    #tic = time.perf_counter()
    s1q = s1 ** 2
    s2q = s2 ** 2
    t1q = t1 ** 2
    t2q = t2 ** 2
    u1q = u1 ** 2
    u2q = u2 ** 2
    #toc = time.perf_counter()
    #print('quad', round(toc - tic, 5))

    #tic = time.perf_counter()
    sq = s ** 2
    tq = t ** 2
    uq = u ** 2

    sq1 = sq[:, :-1]
    sq2 = sq[:, 1:]
    tq1 = tq[:, :-1]
    tq2 = tq[:, 1:]
    uq1 = uq[:, :-1]
    uq2 = uq[:, 1:]

    s1w = cp.repeat(sq1, shape[1] * shape[2])
    s2w = cp.repeat(sq2, shape[1] * shape[2])
    t1w = cp.repeat(tq1, shape[2])
    t2w = cp.repeat(tq2, shape[2])
    t1w = cp.tile(t1w, shape[0])
    t2w = cp.tile(t2w, shape[0])
    u1w = cp.tile(uq1, shape[0] * shape[1])
    u2w = cp.tile(uq2, shape[0] * shape[1])

    n = len(s1w) // n_pt  # 3
    ss = s1w.strides[0]
    s1w = as_strided(s1w, shape=(n, n_pt), strides=(ss, n * ss)).reshape(-1)
    s2w = as_strided(s2w, shape=(n, n_pt), strides=(ss, n * ss)).reshape(-1)
    t1w = as_strided(t1w, shape=(n, n_pt), strides=(ss, n * ss)).reshape(-1)
    t2w = as_strided(t2w, shape=(n, n_pt), strides=(ss, n * ss)).reshape(-1)
    u1w = as_strided(u1w, shape=(n, n_pt), strides=(ss, n * ss)).reshape(-1)
    u2w = as_strided(u2w, shape=(n, n_pt), strides=(ss, n * ss)).reshape(-1)


    toc = time.perf_counter()
    print('quad_new', round(toc - tic, 5))



    tic = time.perf_counter()
    r111 = cp.sqrt(s1q + t1q + u1q)
    r211 = cp.sqrt(s2q + t1q + u1q)
    r121 = cp.sqrt(s1q + t2q + u1q)
    r221 = cp.sqrt(s2q + t2q + u1q)
    r112 = cp.sqrt(s1q + t1q + u2q)
    r212 = cp.sqrt(s2q + t1q + u2q)
    r122 = cp.sqrt(s1q + t2q + u2q)
    r222 = cp.sqrt(s2q + t2q + u2q)
    toc = time.perf_counter()
    print('sqrt', round(toc - tic, 5))






    A1 = arctg(s1, s2, t1, t2, u1, r121, r211)
    A2= -arctg(s1, s2, t2, t1, u1, r111, r221)
    A3 = arctg(s1, s2, t2, t1, u2, r112, r222)
    A4= -arctg(s1, s2, t1, t2, u2, r122, r212)
    B1 = arctg(t2, t1, s2, s1, u1, r121, r211)
    B2= -arctg(t1, t2, s2, s1, u1, r111, r221)
    B3 = arctg(t1, t2, s2, s1, u2, r112, r222)
    B4= -arctg(t2, t1, s2, s1, u2, r122, r212)

    Vrr = -(A1 + A2 + A3 + A4)
    Vzz = B1 + B2 + B3 + B4 - Vrr

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
    V = cp.stack([Vrr, Vxy, Vxz, Vyz, Vzz])

    FB = d @ V

    FB = FB[0].reshape((n_objs, batches))
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