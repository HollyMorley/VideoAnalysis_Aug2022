import sys, os, cv2
import numpy as np

from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pycalib.plot import plotCamera
from pycalib.ba import bundle_adjustment, encode_camera_param, decode_camera_param, make_mask
from pycalib.calib import lookat, triangulate, triangulate_Npts



# -----------------------------------------------------------------------------

# 3D points
# X_gt = (np.random.rand(16, 3) - 0.5)*5 # random points centered at [0, 0, 0]
X_gt = np.array(np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))).reshape(
    (3, -1)).T  # 3D grid points
Np = X_gt.shape[0]
print('X_gt:', X_gt.shape)

# Camera intrinsics
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]).astype(np.float64)  # VGA camera

# Camera poses: cameras are at the vertices of a hexagon
t = 2 * np.pi / 5 * np.arange(5)
v_gt = np.vstack((10 * np.cos(t), 10 * np.sin(t), np.zeros(t.shape))).T
Nc = v_gt.shape[0]
R_gt = []
t_gt = []
P_gt = []
rvec_gt = []
for i in range(Nc):
    t = v_gt[i, :]
    R, t = lookat(t, np.zeros(3), np.array([0, 1, 0]))
    R_gt.append(R)
    t_gt.append(t)
    P_gt.append(K @ np.hstack((R, t)))
    rvec_gt.append(cv2.Rodrigues(R)[0])
R_gt = np.array(R_gt)
t_gt = np.array(t_gt)
P_gt = np.array(P_gt)
rvec_gt = np.array(rvec_gt)
print('R_gt:', R_gt.shape)
print('t_gt:', t_gt.shape)
print('P_gt:', P_gt.shape)
print('rvec_gt:', rvec_gt.shape)

# 2D observations points
x_gt = []
for i in range(Nc):
    xt = cv2.projectPoints(X_gt.reshape((-1, 1, 3)), rvec_gt[i], t_gt[i], K, None)[0].reshape((-1, 2))
    x_gt.append(xt)
x_gt = np.array(x_gt)
print('x_gt:', x_gt.shape)

# Verify triangulation
Y = []
for i in range(Np):
    y = triangulate(x_gt[:, i, :].reshape((-1, 2)), P_gt)
    # print(y)
    Y.append(y)
Y = np.array(Y).T
Y = Y[:3, :] / Y[3, :]
assert np.allclose(0, X_gt - Y.T)

# Verify z > 0 at each camera
for i in range(Nc):
    Xc = R_gt[i] @ X_gt.T + t_gt[i]
    assert np.all(Xc[2, :] > 0)

# Inject gaussian noise to the inital guess
R_est = R_gt.copy()
t_est = t_gt.copy()
K_est = np.array([K for c in range(Nc)])
X_est = X_gt.copy()
x_est = x_gt.copy()

for i in range(Nc):
    R_est[i] = cv2.Rodrigues(cv2.Rodrigues(R_est[i])[0] + np.random.normal(0, 0.01, (3, 1)))[0]
    t_est[i] += np.random.normal(0, 0.01, (3, 1))
    K_est[i][0, 0] = K_est[i][1, 1] = K_est[i][0, 0] + np.random.normal(0, K_est[i][0, 0] / 10)

X_est += np.random.normal(0, 0.01, X_est.shape)
x_est += np.random.normal(0, 0.1, x_est.shape)

# -----------------------------------------------------------------------------

# verify that the output is correct
np.allclose(triangulate_Npts(x_gt, P_gt), X_gt)

# -----------------------------------------------------------------------------

def triangulate_by_loop(x, P):
    X = []
    Np = x.shape[1]
    for i in range(Np):
        y = triangulate(x[:,i,:].reshape((-1,2)), P[:])
        X.append(y)
    X = np.array(X)
    return X[:,:3]

# verify that the output is correct
np.allclose(triangulate_by_loop(x_gt, P_gt), X_gt)