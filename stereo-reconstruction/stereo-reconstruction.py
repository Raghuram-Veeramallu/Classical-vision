import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


def compute_F(pts1, pts2):

    ransac_thr = 1
    ransac_iter = 50
    
    best_fundamental = None
    best_model_inliers = 0

    for i in range(0, ransac_iter):
        # Choosing 8 random indexes to choose 8 random points
        np.random.seed(85)
        idx = np.random.choice(pts1.shape[0], size=8, replace=False)
        u = pts1[idx]
        v = pts2[idx]

        # 8 x 9
        uv = np.array([
            [u[0][0] * v[0][0], u[0][1] * v[0][0], v[0][0], u[0][0] * v[0][1], u[0][1] * v[0][1], v[0][1], u[0][0], u[0][1], 1],
            [u[1][0] * v[1][0], u[1][1] * v[1][0], v[1][0], u[1][0] * v[1][1], u[1][1] * v[1][1], v[1][1], u[1][0], u[1][1], 1],
            [u[2][0] * v[2][0], u[2][1] * v[2][0], v[2][0], u[2][0] * v[2][1], u[2][1] * v[2][1], v[2][1], u[2][0], u[2][1], 1],
            [u[3][0] * v[3][0], u[3][1] * v[3][0], v[3][0], u[3][0] * v[3][1], u[3][1] * v[3][1], v[3][1], u[3][0], u[3][1], 1],
            [u[4][0] * v[4][0], u[4][1] * v[4][0], v[4][0], u[4][0] * v[4][1], u[4][1] * v[4][1], v[4][1], u[4][0], u[4][1], 1],
            [u[5][0] * v[5][0], u[5][1] * v[5][0], v[5][0], u[5][0] * v[5][1], u[5][1] * v[5][1], v[5][1], u[5][0], u[5][1], 1],
            [u[6][0] * v[6][0], u[6][1] * v[6][0], v[6][0], u[6][0] * v[6][1], u[6][1] * v[6][1], v[6][1], u[6][0], u[6][1], 1],
            [u[7][0] * v[7][0], u[7][1] * v[7][0], v[7][0], u[7][0] * v[7][1], u[7][1] * v[7][1], v[7][1], u[7][0], u[7][1], 1],
        ])

        z = null_space(uv)[:, 0] # taking the first solution
        z = z.reshape(3, 3)

        # SVD cleanup
        U, D, V_t = np.linalg.svd(z)
        # D_tilde is D with 0 in the last position (Rank 2)
        D_tilde = D
        D_tilde[2] = 0

        # Fundamental matrix by multiplying them back
        funda_trans = np.dot(U * D_tilde, V_t)

        if (np.linalg.matrix_rank(funda_trans) != 2):
            raise Exception('Rank of fundamental matrix not 2')

        if best_fundamental is None:
            best_fundamental = funda_trans

        inliers_count = 0
        for all_idx in range(0, pts1.shape[0]):
            
            pt1 = np.array([pts1[all_idx][0], pts1[all_idx][1], 1])
            pt2T = np.array([pts2[all_idx][0], pts2[all_idx][1], 1])

            err = np.dot(np.matmul(pt2T, funda_trans), pt1)

            if (abs(err) < ransac_thr):
                inliers_count = inliers_count + 1

        if inliers_count > best_model_inliers:
            best_fundamental = funda_trans
            best_model_inliers = inliers_count
    
    F = np.array(best_fundamental)
    print(best_model_inliers)
    return F


def triangulation(P1, P2, pts1, pts2):
    
    pts3D = []
    for pt1, pt2 in zip(pts1, pts2):
        U = np.array([pt1[0], pt1[1], 1])
        V = np.array([pt2[0], pt2[1], 1])
        U_x = np.array([
            [0, -U[2], U[1]],
            [U[2], 0, -U[0]],
            [-U[1], U[0], 0]
        ])
        V_x = np.array([
            [0, -V[2], V[1]],
            [V[2], 0, -V[0]],
            [-V[1], V[0], 0]
        ])

        A_u = np.matmul(U_x, P1)
        A_v = np.matmul(V_x, P2)

        A = np.concatenate((A_u, A_v), axis=0)

        _, _, v = np.linalg.svd(A)

        vl = v[-1, :]
        pts_each = [
            vl[0]/vl[3],
            vl[1]/vl[3],
            vl[2]/vl[3],
        ]
        pts3D.append(pts_each)

    pts3D = np.array(pts3D)

    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    chir_chk_counts = []
    for r, c, x in zip(Rs, Cs, pts3Ds):
        # last row of R (r3)
        chir_check = np.matmul((x - c.reshape(-1)), r[2, :])
        check_count = len(np.where(chir_check > 0)[0])
        chir_chk_counts.append(check_count)
    idx = np.argmax(chir_chk_counts)
    R = Rs[idx]
    C = Cs[idx]
    pts3D = pts3Ds[idx]

    return R, C, pts3D


def compute_rectification(K, R, C):
    
    C = C.reshape(1, 3)
    r1 = C / np.linalg.norm(C)
    Cx = C[0, 0]
    Cy = C[0, 1]
    r2 = [-Cy, Cx, 0]/(np.sqrt(Cx ** 2 + Cy ** 2))
    r2 = r2.reshape(1, 3)
    r3 = np.cross(r1, r2)
    R_rect = np.concatenate((r1, r2, r3), axis=0)

    KR_rect = np.matmul(K, R_rect)

    H1 = np.matmul(KR_rect, np.linalg.inv(K))
    H2 = np.matmul(np.matmul(KR_rect, R.T), np.linalg.inv(K))

    return H1, H2


def dense_match(img1, img2, descriptors1, descriptors2):
    disparity = np.zeros(img1.shape)
    img_h, img_w = img1.shape
    for i in range(0, img_h):
        nn = NearestNeighbors(n_neighbors=1).fit(descriptors1[i, :, :])
        _, idxs = nn.kneighbors(descriptors2[i, :, :])
        # nearest corresponding point index computation
        shift = np.argmin(np.abs(idxs - i), axis=1)
        final_idx = idxs[range(len(shift)), shift]
        disparity[i, :] = abs(final_idx - np.arange(img_w))

    # not having disparity for the black space (pix intensity 0)
    disparity = np.where(img1 == 0, 0, disparity)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = (F @ np.array([[p[0], p[1], 1]]).T).flatten()
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    disparity[disparity > 150] = 150
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./stereo-reconstruction/left.bmp', 1)
    img_right = cv2.imread('./stereo-reconstruction/right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 0: get correspondences between image pair
    data = np.load('./stereo-reconstruction/resource/correspondence.npz')
    pts1, pts2 = data['pts1'], data['pts2']
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 1: compute fundamental matrix and recover four sets of camera poses
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 2: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 3: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 4: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 5: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    data = np.load('./stereo-reconstruction/resource/dsift_descriptor.npz')
    desp1, desp2 = data['descriptors1'], data['descriptors2']
    disparity = dense_match(img_left_w, img_right_w, desp1, desp2)
    visualize_disparity_map(disparity)
