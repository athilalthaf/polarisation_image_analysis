"""
Implementing polarisation moadel based on "An Analytical Model for Skylight Polarisation" Wilkie. et al. 2004

implementation is not complete
"""



from lib_importer import *


gamma = np.linspace(0.00001, 2 * np.pi, 200) # scattering angle
gamma_ang = np.linspace(0, 360, 200, dtype=int) #scattering angle space
Degree_of_pol = np.sin(gamma) ** 2 / (1 + (np.cos(gamma)) ** 2) #degree of polarisation based on scattering angle

Turbidity = 1 #atmospheric turbidity

theta_sun = np.deg2rad(60) # position of sun
gamma_sun = np.deg2rad(0.00001)
azi_sun = np.deg2rad(0.00001)


Y_dist_values = np.array([[0.1787, -1.4630],[-0.3554, 0.4275],[-0.0227, 5.3251],[0.1206, -2.5771],[-0.0670, 0.3703]])
Y_dist = np.matmul(Y_dist_values,np.matrix([Turbidity,1]).T)

x_dist_values = np.array([[-0.0193, -0.2592], [-0.0665, 0.0008], [-0.0004, 0.2125], [-0.0641, -0.8989], [-0.0033, 0.0452]])
x_dist = np.matmul(x_dist_values, np.matrix([Turbidity, 1]).T)

y_dist_values = np.array([[-0.0167, -0.2608], [-0.0950, 0.0092], [-0.0079, 0.2102], [-0.0441, -1.6537], [-0.0109, 0.0529]])
y_dist = np.matmul(y_dist_values, np.matrix([Turbidity, 1]).T)
C = 1.2 # scaling factor



def Perez_luminance(gamma, theta):
    """

    :param gamma: float
        scattering angle
    :param theta: float
        azimuth
    :return: float
        perez luminance to the corresponding inputs
    """
    # A = np.linalg.norm([x_dist[0], y_dist[0], Y_dist[0]])
    # B = np.linalg.norm([x_dist[1], y_dist[1], Y_dist[1]])
    # C = np.linalg.norm([x_dist[2], y_dist[2], Y_dist[2]])
    # D = np.linalg.norm([x_dist[3], y_dist[3], Y_dist[3]])
    # E = np.linalg.norm([x_dist[4], y_dist[4], Y_dist[4]])
    A,B,C,D,E = 1, -1, 1, -1, 1
    return (1 + A * np.exp(B/np.cos(theta))) * (1 + C * np.exp(D * gamma) + E * np.cos(gamma)**2)

def lin_pol(gamma):
    """
    :param gamma: float
        scattering angle
    :return: int
        linear polarisation corresponding to that angle
    """
    return np.sin(gamma)**2 /(1 + np.cos(gamma)**2)


I_sun = Perez_luminance(gamma_sun,theta_sun) #sun luminance
I_90 = Perez_luminance(gamma,theta_sun)

def skylight_intensity(gamma,theta):
    """
    skylight intensity function
    :param gamma: float
        scattering angle of the light
    :param theta:  float
        azimuth position
    :return: float
        skylight intensity function
    """
    return (1/Perez_luminance(gamma, theta) - 1/I_sun) * I_90 * I_sun / (I_sun - I_90)

def Polarisation(gamma,theta):
    """
    Polarisation function with scattering
    :param gamma: float
        scattering angle
    :param theta: float
        azimuthal position
    :return: float
        polarisation pattern
    """
    return 1/C * lin_pol(gamma) * (theta * np.cos(theta) + (np.pi/2 - theta) * skylight_intensity(gamma, theta))

sun_theta = 0.001
sun_gamma = 0
sun_phi = np.pi/3
gamma_range = np.deg2rad(np.linspace(0, 360, 200))
theta_range = np.deg2rad(np.linspace(0.0001, 90, 100))

gamma_mesh, theta_mesh = np.meshgrid(gamma_range, theta_range)

Perez_test = Perez_luminance(gamma_mesh, theta_mesh)

Chi = (4/9 - Turbidity/120) * (np.pi - 2 * theta_sun)

Y_z = (4.0453 * Turbidity - 4.9710) * np.tan(Chi) - 0.2155 * Turbidity + 2.4192
Y = Y_z * Perez_luminance(gamma_mesh, theta_mesh)/Perez_luminance(0, theta_sun)


# plt.subplot(111)
# plt.plot(gamma, Perez_luminance(theta=np.pi/2, gamma=gamma))
# plt.xlabel("scattering angle")
# plt.ylabel("Luminance")
# plt.title("Scattering angle v/s Luminance")
# plt.show()

fig, ax = plt.subplots(dpi=120,subplot_kw=dict(projection="polar"),constrained_layout=True)
plt1 = ax.contourf(gamma_mesh, theta_mesh, Perez_test, 100, cmap="inferno")
plt.title("Perez Luminance based on the scatter angle and elevation")
fig.colorbar(plt1)
ax.set_yticks([np.deg2rad(30), np.deg2rad(60)])
ax.yaxis.set_ticklabels([30, 60])
# plt.ylim([90, 0])
plt.show()

fig2, ax2 = plt.subplots(dpi=120, subplot_kw=dict(projection="polar"), constrained_layout=True)
plt2 = ax2.contourf(gamma_mesh, theta_mesh, Y, 100, cmap="inferno")
plt.title("Luminance Based on Zenith luminance")
# ax2.scatter(sun_gamma, sun_theta)
fig2.colorbar(plt2)
ax2.set_yticks([np.deg2rad(30), np.deg2rad(60)])
ax2.yaxis.set_ticklabels([30, 60])
# plt.ylim([90, 0])
plt.show()

#
# pol = Polarisation(gamma_mesh,theta_mesh)
# fig3, ax3 = plt.subplots(dpi=120,subplot_kw=dict(projection="polar"))
# ax2.contourf(gamma_mesh, theta_mesh, pol, 100, cmap="inferno")
# plt.title("Polarisation pattern based on the scatter angle and elevation including sun_position")
# # ax2.scatter(sun_gamma, sun_theta)
# # ax2.set_yticks([np.deg2rad(30), np.deg2rad(60)])
# # ax2.yaxis.set_ticklabels([30, 60])

plt.show()

fig4, ax4 = plt.subplots(dpi=120,subplot_kw=dict(projection="polar"),constrained_layout=True)
plt4 = ax4.contourf(gamma_mesh, theta_mesh, Polarisation(gamma_mesh, theta_mesh), 100, cmap="inferno")
plt.title("Polarisation pattern based on the scatter angle and elevation")
fig4.colorbar(plt4)
ax4.set_yticks([np.deg2rad(30), np.deg2rad(60)])
ax4.yaxis.set_ticklabels([30, 60])
# plt.ylim([90, 0])
plt.show()

# fig5, ax5 = plt.subplots(dpi=120, constrained_layout=True)
# plt5 = ax5.plot(gamma_mesh, lin_pol(gamma_mesh))
# plt.title("Polarisation pattern based on the scatter angle and elevation")
# # ax4.set_xticks([0, np.deg2rad(90), np.deg2rad(180), np.deg2rad(360)])
# # tick = np.arange(0,361,90)
# # ax4.set_xticks(np.deg2rad(tick))
# #
# # ax4.xaxis.set_ticklabels(tick)
# # plt.ylim([90, 0])
# plt.show()
#
