import numpy as np
import matplotlib.pyplot as plt

gamma = np.linspace(0, 2 * np.pi, 200) # scattering angle
gamma_ang = np.linspace(0, 360, 200, dtype=int)
Degree_of_pol = np.sin(gamma) ** 2 / (1 + (np.cos(gamma)) ** 2)
ax = plt.figure().add_subplot(111)
Turbidity = 1
Y_dist_values = np.array([[0.1787,-1.4630],[-0.3554,0.4275],[-0.0227, 5.3251],[0.1206, -2.5771],[-0.0670, 0.3703]])
Y_dist = np.matmul(Y_dist_values,np.matrix([Turbidity,1]).T)

x_dist_values = np.array([[-0.0193, -0.2592], [-0.0665, 0.0008], [-0.0004, 0.2125], [-0.0641, -0.8989], [-0.0033, 0.0452]])
x_dist = np.matmul(x_dist_values,np.matrix([Turbidity,1]).T)

y_dist_values = np.array([[-0.0167, -0.2608], [-0.0950, 0.0092], [-0.0079, 0.2102], [-0.0441, -1.6537], [-0.0109, 0.0529]])
y_dist = np.matmul(y_dist_values, np.matrix([Turbidity, 1]).T)

def Perez_luminance(theta,gamma ,A =1,B=1,C =1, D=1,E =1):
    return (1 + A * np.exp(B/np.cos(theta))) * (1 + C * np.exp(D * gamma) + E * np.cos(gamma)**2)

sun_theta, sun_gamma = np.pi/3, 0.001
plt.subplot(111)
plt.plot(gamma, Perez_luminance(theta=1, gamma=gamma))
plt.show()


# plt.subplot(111)
# plt.plot(gamma, Degree_of_pol)
# plt.xlim([0, 2 * np.pi])
# plt.ylim([0, 1])

# ax.xaxis.set_ticks(theta_ang)
# a = ax.get_xticks().tolist()
# a = gamma_ang
# ax.set_xticklabels(a)
plt.xlabel("scattering angle")
plt.ylabel("degree of polarisation")
plt.title("Scattering angle v/s degree of polarisation")
plt.show()
