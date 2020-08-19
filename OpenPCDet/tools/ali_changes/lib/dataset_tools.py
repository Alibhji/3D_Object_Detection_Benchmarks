from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.

    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


def draw_point_cloud(data , ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    pp = copy.deepcopy(data['points'])
    points = 0.2
    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, pp.shape[0], points_step)

    # colors = {
    #     'Car': 'b',
    #     'Tram': 'r',
    #     'Cyclist': 'g',
    #     'Van': 'c',
    #     'Truck': 'm',
    #     'Pedestrian': 'y',
    #     'Sitter': 'k'
    # }
    colors = ['k','r','y','g']

    axes_limits = [
        [-20, 80],  # X axis range
        [-20, 20],  # Y axis range
        [-3, 10]  # Z axis range
    ]
    axes_str = ['X', 'Y', 'Z']

    ax.scatter(*np.transpose(pp[:, axes]), s=point_size, c=pp[:, 3], cmap='gray')
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d != None:
        ax.set_xlim3d(xlim3d)
    if ylim3d != None:
        ax.set_ylim3d(ylim3d)
    if zlim3d != None:
        ax.set_zlim3d(zlim3d)

    for gt in (data['gt_boxes']):
        x, y, z = gt[0:3]
        l, w, h = gt[3:6]
        cls = int(gt[-1])
        # print("------------------>",cls)
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])

        yaw = gt[6]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])

        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.array([[x, y, z]]).T
        # print(cornerPosInVelo)
        draw_box(ax, cornerPosInVelo, axes=axes, color=colors[cls])