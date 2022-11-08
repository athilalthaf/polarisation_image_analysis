import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
gamma = np.linspace(0, 2 * np.pi, 200) # scattering angle
gamma_ang = np.linspace(0, 360, 200, dtype=int)
Degree_of_pol = np.sin(gamma) ** 2 / (1 + (np.cos(gamma)) ** 2)

Turbidity = 1

theta_sun = np.deg2rad(60)
gamma_sun = np.deg2rad(0)

Y_dist_values = np.array([[0.1787,-1.4630],[-0.3554,0.4275],[-0.0227, 5.3251],[0.1206, -2.5771],[-0.0670, 0.3703]])
Y_dist = np.matmul(Y_dist_values,np.matrix([Turbidity,1]).T)

x_dist_values = np.array([[-0.0193, -0.2592], [-0.0665, 0.0008], [-0.0004, 0.2125], [-0.0641, -0.8989], [-0.0033, 0.0452]])
x_dist = np.matmul(x_dist_values,np.matrix([Turbidity,1]).T)

y_dist_values = np.array([[-0.0167, -0.2608], [-0.0950, 0.0092], [-0.0079, 0.2102], [-0.0441, -1.6537], [-0.0109, 0.0529]])
y_dist = np.matmul(y_dist_values, np.matrix([Turbidity, 1]).T)
C = 1.2 # scaling factor

def Perez_luminance(gamma, theta):
    # A = np.linalg.norm([x_dist[0], y_dist[0], Y_dist[0]])
    # B = np.linalg.norm([x_dist[1], y_dist[1], Y_dist[1]])
    # C = np.linalg.norm([x_dist[2], y_dist[2], Y_dist[2]])
    # D = np.linalg.norm([x_dist[3], y_dist[3], Y_dist[3]])
    # E = np.linalg.norm([x_dist[4], y_dist[4], Y_dist[4]])
    A,B,C,D,E = 1, -1, 1, -1, 1
    return (1 + A * np.exp(B/np.cos(theta))) * (1 + C * np.exp(D * gamma) + E * np.cos(gamma)**2)

def lin_pol(gamma):
    return np.sin(gamma)**2 /(1 + np.cos(gamma)**2)


I_sun = Perez_luminance(gamma_sun,theta_sun)
I_90 = Perez_luminance(gamma,theta_sun)

def skylight_intensity(gamma,theta):
    return (1/Perez_luminance(gamma, theta) - 1/I_sun) * I_90 * I_sun / (I_sun - I_90)

def Polarisation(gamma,theta):
    return 1/C * lin_pol(gamma) * (theta * np.cos(theta) + (np.pi/2 - theta) * skylight_intensity(gamma, theta))


gamma_range = np.deg2rad(np.linspace(0, 360, 200))
theta_range = np.deg2rad(np.linspace(90, 0, 100))

gamma_mesh, theta_mesh = np.meshgrid(gamma_range, theta_range)

Perez_test = Perez_luminance(gamma_mesh, theta_mesh)

Y_z = Perez_luminance(0,0)
Y = Y_z * Perez_luminance(gamma_mesh, theta_mesh)/Perez_luminance(0, theta_sun)

sun_theta = np.pi/3
sun_gamma = 0
sun_phi = np.pi/3
# plt.subplot(111)
# plt.plot(gamma, Perez_luminance(theta=np.pi/2, gamma=gamma))
# plt.xlabel("scattering angle")
# plt.ylabel("Luminance")
# plt.title("Scattering angle v/s Luminance")
# plt.show()

# fig, ax = plt.subplots(dpi=120,subplot_kw=dict(projection="polar"))
# ax.contourf(gamma_mesh, theta_mesh, Perez_test, 100,cmap="inferno")
# plt.title("Perez Luminance based on the scatter angle and elevation")
#
# ax.set_yticks([np.deg2rad(30), np.deg2rad(60)])
# ax.yaxis.set_ticklabels([30, 60])
# # plt.ylim([90, 0])
# plt.show()

fig2, ax2 = plt.subplots(dpi=120,subplot_kw=dict(projection="polar"))
ax2.contourf(gamma_mesh, theta_mesh, Y, 100, cmap="inferno")
plt.title("Perez Luminance based on the scatter angle and elevation including sun_position")
ax2.scatter(sun_gamma,sun_theta)
ax2.set_yticks([np.deg2rad(30), np.deg2rad(60)])
ax2.yaxis.set_ticklabels([30, 60])
# plt.ylim([90, 0])
plt.show()

# plt.subplot(111)
# plt.plot(gamma, Degree_of_pol)
# plt.xlim([0, 2 * np.pi])
# plt.ylim([0, 1])

# ax.xaxis.set_ticks(theta_ang)
# a = ax.get_xticks().tolist()
# a = gamma_ang
# ax.set_xticklabels(a)
# plt.xlabel("scattering angle")
# plt.ylabel("degree of polarisation")
# plt.title("Scattering angle v/s degree of polarisation")
# plt.show()

