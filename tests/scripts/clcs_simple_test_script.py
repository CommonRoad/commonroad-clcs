import numpy as np
from matplotlib import pyplot as plt

from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem


# ref path
x_coords = np.linspace(0, 5, num=11)
y_coords = np.zeros(11)
ref_path = np.column_stack((x_coords, y_coords))

# Ccosy settings
default_lim = 30.0
eps = 0.1
eps2 = 0.3
method = 1

# construct ccosy and get projection domain polygon
ccosy_1 = CurvilinearCoordinateSystem(ref_path, default_lim, eps, eps2, method)
proj_domain_1 = np.asarray(ccosy_1.projection_domain())

# get modified ref path from ccosy
ref_path_ccosy = np.asarray(ccosy_1.reference_path())

# point conversion
pt_cart = np.array([1, 10])
ccosy_1.cartesian_point_inside_projection_domain(pt_cart[0], pt_cart[1])
pt_curv = ccosy_1.convert_to_curvilinear_coords(pt_cart[0], pt_cart[1])
pt_proj = ccosy_1.convert_to_cartesian_coords(pt_curv[0], 0.0)


# ==== Visualization
# plot cartesian proj domains
plt.figure()
plt.title("Cartesian CCosy 1")
plt.plot(ref_path_ccosy[:, 0], ref_path_ccosy[:, 1], color='g', marker='x', markersize=5, zorder=19, linewidth=2.0)
plt.plot(proj_domain_1[:, 0], proj_domain_1[:, 1], zorder=100, marker=".", color='orange')
plt.plot([pt_cart[0], pt_proj[0]], [pt_cart[1], pt_proj[1]], zorder=100, linewidth=2,
         marker='x', markersize=9, color='blue')
plt.show()