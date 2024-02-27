#from utils import labels
import pandas as pd
import numpy as np

def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]

    return ellipse_center(a)

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num

    return np.array([x0, y0])

def mirror_back(backy, backx):
    y2 = backy.loc(axis=1)['Back1'].values
    y1 = backy.loc(axis=1)['Back12'].values
    x2 = backx.loc(axis=1)['Back1'].values
    x1 = backx.loc(axis=1)['Back12'].values

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    intercept = backx * m + b
    diffmirror = intercept.values - backy.values

    mirror = intercept + diffmirror
    mirror.rename(columns={'x': 'yrev'}, inplace=True)

    return mirror

def estimate_body_center(backy, backx):
    # body_labels = labels.extract(tracking_data, 'BodyR', 'BodyL', tuple=True, coords=True)
    # body_labels_x = [x_coord for x_coord in body_labels if 'x' in x_coord]
    # body_labels_y = [y_coord for y_coord in body_labels if 'y' in y_coord]

    mirror = mirror_back(backy, backx)
    combinedy = pd.concat([backy, mirror], axis=1)

    backx_repeat = backx.copy(deep=True)
    backx_repeat.rename(columns={'x': 'xrev'}, inplace=True)
    combinedx = pd.concat([backx, backx_repeat], axis=1)

    centroids = []
    #idxs = []
    for i in range(combinedy.shape[0]):
        #idx = combinedy.index[i]
        x = combinedx.iloc[i]
        y = combinedy.iloc[i]

        if pd.isna(x).all() or pd.isna(y).all():
            centroid = np.array([np.nan, np.nan])
        else:
            centroid = fitEllipse(x.dropna(), y.dropna())

        centroids.append(centroid)
        #idxs.append(idx)
    #body_center = pd.DataFrame(centroids, columns=['x_centroid', 'y_centroid'], index=pd.MultiIndex.from_tuples(idxs, names=('RunStage', 'FrameIdx')))
    body_center = pd.DataFrame(centroids, columns=['x_centroid', 'y_centroid'], index=combinedy.index)
    return body_center



