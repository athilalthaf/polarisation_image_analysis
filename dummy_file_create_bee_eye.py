from invertpy.sense import CompoundEye
from invertpy.io import save_eye2csv, load_csv2eye

from scipy.interpolate import splev, splrep
from scipy.spatial.transform import Rotation as R

import numpy as np

alpha_max_1_2 = np.deg2rad(np.array([
    [
        -74.5, -69.7, -60.2, -50.2, -40.2, -30.3, -20.3, -10.2, 0.0,
        10.2, 20.2, 30.1, 39.9, 49.4, 58.9, 67.8, 69.0
    ], [
        270.0, 191.9, 176.8, 174.6, 173.4, 166.8, 160.5, 158.5, 156.7,
        158.5, 163.4, 174.3, 189.0, 202.4, 215.8, 244.7, 270.0
    ]
]))
alpha_max_3_4 = np.deg2rad(np.array([
    [
        -74.5, -67.8, -58.1, -48.4, -38.9, -29.4, -19.8, -9.9, 0.0,
        9.9, 19.8, 29.6, 39.4, 49.2, 58.9, 68.1, 69.0
    ], [
        -90.0, -63.5, -52.5, -44.1, -36.1, -26.8, -20.6, -14.6, -14.5, -14.4,
        -15.5, -21.2, -25.5, -31.6, -42.0, -70.7, -90.0
    ]
]))
tck_1_2 = splrep(alpha_max_1_2[0], alpha_max_1_2[1], s=0.01)
tck_3_4 = splrep(alpha_max_3_4[0], alpha_max_3_4[1], s=0.01)


def build_right_bee_eye():
    """
    Following [1]_

    Notes
    -----
    .. [1] "Mimicking honeybee eyes with a 280 degrees field of view catadioptric imaging system"
            http://iopscience.iop.org/article/10.1088/1748-3182/5/3/036002/pdf
    """
    Delta_phi_min_v = np.deg2rad(1.5)
    Delta_phi_max_v = np.deg2rad(4.5)
    Delta_phi_min_h = np.deg2rad(2.4)
    Delta_phi_mid_h = np.deg2rad(3.7)
    Delta_phi_max_h = np.deg2rad(4.6)

    def norm_alpha(alpha):
        alpha = alpha % (2 * np.pi)
        if alpha > 3 * np.pi / 2:
            alpha -= 2 * np.pi
        return alpha

    def norm_epsilon(epsilon):
        epsilon = epsilon % (2 * np.pi)
        if epsilon > np.pi:
            epsilon -= 2 * np.pi
        return epsilon

    def xi(z=None, alpha=None, epsilon=None):
        if alpha is not None:
            if z in [1, 2]:  # or 0 <= alpha <= 3 * np.pi / 2:
                return 1.
            if z in [3, 4]:  # or 3 * np.pi / 2 <= alpha:
                return -1.
        if epsilon is not None:
            if z in [1, 4]:  # or 0 <= epsilon <= np.pi / 2:
                return 1.
            if z in [2, 3]:  # or 3 * np.pi / 2 <= epsilon:
                return -1.
        return 0.

    def Delta_alpha(Delta_phi_h, epsilon):
        return norm_alpha((2 * np.arcsin(np.sin(Delta_phi_h / 2.) / np.cos(epsilon))) % (2 * np.pi))

    def phi_h(z, alpha, epsilon):
        alpha = norm_alpha(alpha)
        epsilon = norm_epsilon(epsilon)
        abse = np.absolute(epsilon)
        absa = np.absolute(alpha)

        # zone Z=1/2
        if z in [1, 2]:
            if 0 <= alpha <= np.pi / 4:
                return Delta_phi_mid_h + alpha / (np.pi / 4) * (Delta_phi_min_h - Delta_phi_mid_h)
            if np.pi / 4 < alpha <= np.pi / 2:
                return Delta_phi_min_h + (alpha - np.pi / 4) / (np.pi / 4) * (Delta_phi_mid_h - Delta_phi_min_h)
            if np.pi / 2 < alpha <= np.deg2rad(150) and abse <= np.deg2rad(50):
                return Delta_phi_mid_h + (alpha - np.pi / 2) / (np.pi / 3) * (Delta_phi_max_h - Delta_phi_mid_h)
            if np.pi / 2 < alpha <= np.deg2rad(150) and abse > np.deg2rad(50):
                return Delta_phi_mid_h + (alpha - np.pi / 2) / (np.pi / 3) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                             * (np.pi / 2 - abse) / np.deg2rad(40)
            if np.deg2rad(150) < alpha <= np.pi and abse <= np.deg2rad(50):
                return Delta_phi_max_h
            if np.deg2rad(150) < alpha <= np.pi and abse > np.deg2rad(50):
                return Delta_phi_mid_h + (Delta_phi_max_h - Delta_phi_mid_h) * (np.pi / 2 - abse) / np.deg2rad(40)
            if np.pi < alpha <= 3 * np.pi / 2:
                return Delta_phi_max_h

        if z in [3, 4]:
            # zone Z=3
            if -np.pi / 2 <= alpha < -np.pi / 4 and epsilon <= -np.deg2rad(50):
                return Delta_phi_mid_h + (absa - np.pi / 4) / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                         * (np.pi / 2 - abse) / np.deg2rad(40)
            if -np.pi / 4 <= alpha <= 0 and epsilon <= -np.deg2rad(50):
                return Delta_phi_mid_h
            if -np.pi / 4 < alpha <= 0 and -np.deg2rad(50) < epsilon <= 0:
                return Delta_phi_mid_h + absa / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                         * (np.deg2rad(50) - abse) / np.deg2rad(50)

            # zone Z=4
            if -np.pi / 2 <= alpha <= -np.pi / 4 and epsilon >= np.deg2rad(50):
                return Delta_phi_max_h + (Delta_phi_max_h - Delta_phi_mid_h) * (np.pi / 2 - abse) / np.deg2rad(40)
            if -np.pi / 4 <= alpha <= 0 and 0 < epsilon <= np.deg2rad(50):
                return Delta_phi_mid_h + absa / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h)
            if -np.pi / 4 <= alpha <= 0 and epsilon >= np.deg2rad(50):
                return Delta_phi_mid_h + absa / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                         * (np.pi / 2 - epsilon) / np.deg2rad(40)
        return np.nan

    def alpha_max(z, epsilon):
        if z in [1, 2]:
            return splev(epsilon, tck_1_2, ext=3, der=0)
        if z in [3, 4]:
            return splev(epsilon, tck_3_4, ext=3, der=0)
        return np.nan

    ommatidia = []
    for z in range(1, 5):
        j = 0
        epsilon = 0.  # elevation
        alpha = 0.  # azimuth
        while np.absolute(epsilon) <= np.pi / 2:
            if j % 2 == 0:
                alpha = 0
            else:
                alpha = Delta_alpha(xi(z=z, alpha=alpha) * Delta_phi_mid_h / 2, epsilon)
            while np.absolute(alpha) < np.absolute(alpha_max(z, epsilon)):
                # print z, np.rad2deg(epsilon), np.rad2deg(alpha)
                Delta_phi_h = phi_h(z, alpha, epsilon)
                if np.isnan(Delta_phi_h):
                    break
                ommatidia.append(np.array([alpha, epsilon]))
                alpha = norm_alpha(alpha + Delta_alpha(xi(z=z, alpha=alpha) * Delta_phi_h, epsilon))
            Delta_phi_v = Delta_phi_min_v + (Delta_phi_max_v - Delta_phi_min_v) * np.absolute(epsilon) / (np.pi / 2)
            epsilon = norm_epsilon(epsilon + xi(z=z, epsilon=epsilon) * Delta_phi_v / 2.)
            j += 1

    ommatidia = np.array(ommatidia)
    omm_ori = R.from_euler('ZY', ommatidia)
    omm_rho = np.deg2rad(5) * (1 + ((np.pi/2 - ommatidia[:, 1]) % (2 * np.pi)) / np.pi)
    omm_pol = np.asarray(ommatidia[:, 1] > np.pi/3, dtype=float)
    spectral = (omm_pol[..., np.newaxis] * np.array([[0, 0, 0, 0, 1]], dtype=float) +
                (1 - omm_pol)[..., np.newaxis] * np.array([[0, 0, 1, 0, 0]], dtype=float))

    return CompoundEye(omm_ori=omm_ori, omm_rho=omm_rho, omm_pol_op=omm_pol, c_sensitive=spectral, name='bee_right_eye')


def main(*args):
    r_eye = build_right_bee_eye()
    print(r_eye)
    l_eye = CompoundEye.flip(r_eye, horizontally=True, name='bee_left_eye')
    print(l_eye)

    save_eye2csv(r_eye, 'bee_right')
    save_eye2csv(l_eye, 'bee_left')

    import matplotlib.pyplot as plt

    hue = l_eye.hue_sensitive
    rgb = hue[..., 1:4]
    rgb[:, [0, 2]] += hue[..., 4:5] / 2
    rgb[:, 0] += hue[..., 0]
    plt.subplot(111, polar=False)
    yaw, pitch, raw = r_eye.omm_ori.as_euler('ZYX', degrees=True).T
    plt.scatter(yaw, pitch, s=20, c=np.clip(rgb, 0, 1))
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.show()


if __name__ == '__main__':
    import sys

    main(*sys.argv)
