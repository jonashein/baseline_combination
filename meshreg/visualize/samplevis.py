import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from libyana.visutils.viz2d import visualize_joints_2d

from meshreg.datasets.queries import BaseQueries, TransQueries
from meshreg.visualize import consistdisplay


def get_check_none(data, key, cpu=True):
    if key in data and data[key] is not None:
        if cpu:
            return data[key].cpu().detach()
        else:
            return data[key].detach().cuda()
    else:
        return None

def _transform(points3d, Rt):
    # points3d: (B,N,3)
    # Rt: (B,3,4)
    hom_points3d = np.concatenate([points3d, np.ones([points3d.shape[0], points3d.shape[1], 1])], axis=2)
    trans_points3d = hom_points3d @ Rt.transpose((0,2,1))
    return trans_points3d

def compute_confidence_ellipses(mean, cov, n_std=2):
    # Based on https://stackoverflow.com/a/20127387
    # Make sure inputs are batched, add batch dimension otherwise
    mean = mean.numpy()
    cov = cov.numpy()
    if mean.ndim == 1:
        mean = np.expand_dims(mean, axis=0)
    if cov.ndim == 2:
        cov = np.expand_dims(cov, axis=0)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.sqrt(eigvals)

    res = []
    for i in range(mean.shape[0]):
        e = Ellipse(xy=tuple(mean[i]),
                    width=eigvals[i, 0] * n_std * 2,
                    height=eigvals[i, 1] * n_std * 2,
                    angle=np.rad2deg(np.arccos(eigvecs[i, 0, 0])),
                    facecolor=(1.0, 0.0, 0.0, 0.2),
                    edgecolor=(1.0, 0.0, 0.0, 0.5))
        res.append(e)
    return res


# def compute_gt_drill_angles(sample):
#     gt_pose = get_check_none(sample, BaseQueries.OBJPOSE)
#
#     if gt_pose is not None:
#         if isinstance(gt_pose, torch.Tensor):
#             gt_pose = gt_pose.numpy()
#
#         # Compute drill bit position, orientation
#         # THIS IS ONLY VALID FOR OUR EXACT DRILL MODEL!
#         DRILL_TIP = np.array([0.053554, 0.225361, -0.241646])
#         DRILL_SHANK = np.array([0.057141, 0.220794, -0.121545])
#         batched_drill_tip = np.tile(np.expand_dims(DRILL_TIP, axis=(0, 1)), (gt_pose.shape[0], 1, 1))
#         batched_drill_shank = np.tile(np.expand_dims(DRILL_SHANK, axis=(0, 1)), (gt_pose.shape[0], 1, 1))
#
#         gt_drill_tip = _transform(batched_drill_tip, gt_pose)
#         gt_drill_shank = _transform(batched_drill_shank, gt_pose)
#         gt_drill_vec = gt_drill_tip - gt_drill_shank
#         gt_drill_vec = gt_drill_vec / np.expand_dims(np.linalg.norm(gt_drill_vec, axis=2), axis=1)
#
#         # Compute horizonal angle (projected onto the YZ plane)
#         z_axis_2d = np.array([0.0, 1.0]).reshape(1, 2, 1)
#         gt_drill_rot_X = np.rad2deg(np.arccos(gt_drill_vec[:, :, [2,0]] @ z_axis_2d))
#         gt_drill_rot_X = np.squeeze(gt_drill_rot_X, axis=(1,2))
#         # Compute vertical angle (projected onto the XZ plane)
#         gt_drill_vec_invY = np.stack([gt_drill_vec[:, :, 2], -gt_drill_vec[:, :, 1]], axis=-1)
#         gt_drill_rot_Y = np.rad2deg(np.arccos(gt_drill_vec_invY @ z_axis_2d))
#         gt_drill_rot_Y = np.squeeze(gt_drill_rot_Y, axis=(1, 2))
#
#         return gt_drill_rot_X, gt_drill_rot_Y, gt_drill_vec
#     return None, None, None


def sample_vis(sample, results, save_img_path, fig=None, max_rows=5, display_centered=False):
    fig.clf()
    images = sample[TransQueries.IMAGE].permute(0, 2, 3, 1).cpu() + 0.5
    batch_size = images.shape[0]
    # pred_handverts2d = get_check_none(results, "verts2d")
    gt_objverts2d = get_check_none(sample, TransQueries.OBJVERTS2D)
    pred_objverts2d = get_check_none(results, "obj_verts2d")
    gt_objcorners2d = get_check_none(sample, TransQueries.OBJCORNERS2D)
    pred_objcorners2d = get_check_none(results, "obj_corners2d")
    gt_objcorners3dw = get_check_none(sample, BaseQueries.OBJCORNERS3D)
    pred_objcorners3d = get_check_none(results, "obj_corners3d")
    gt_objverts3d = get_check_none(sample, TransQueries.OBJVERTS3D)
    pred_objverts3d = get_check_none(results, "obj_verts3d")
    gt_canobjverts3d = get_check_none(sample, TransQueries.OBJCANROTVERTS)
    pred_objverts3dw = get_check_none(results, "recov_objverts3d")
    gt_canobjcorners3d = get_check_none(sample, TransQueries.OBJCANROTCORNERS)
    pred_objcorners3dw = get_check_none(results, "recov_objcorners3d")
    gt_handjoints2d = get_check_none(sample, TransQueries.JOINTS2D)
    pred_handjoints2d = get_check_none(results, "joints2d")
    gt_handjoints3d = get_check_none(sample, TransQueries.JOINTS3D)
    pred_handjoints3d = get_check_none(results, "joints3d")
    gt_handverts3d = get_check_none(sample, TransQueries.HANDVERTS3D)
    pred_handverts3d = get_check_none(results, "verts3d")
    gt_objverts3dw = get_check_none(sample, BaseQueries.OBJVERTS3D)
    pred_handjoints3dw = get_check_none(results, "recov_joints3d")
    gt_handjoints3dw = get_check_none(sample, BaseQueries.JOINTS3D)
    pred_objfps2d = get_check_none(results, "kpt_2d")
    gt_objfps2d = get_check_none(sample, BaseQueries.OBJFPS2D)
    pred_objvar2d = get_check_none(results, "var")
    # gt_drill_angle_X, gt_drill_angle_Y, tmp = compute_gt_drill_angles(sample)

    row_nb = min(max_rows, batch_size)
    if display_centered:
        col_nb = 7
    else:
        col_nb = 5
    axes = fig.subplots(row_nb, col_nb)

    for row_idx in range(row_nb):
        # Column 0
        col_idx = 0
        axes[row_idx, col_idx].imshow(images[row_idx])
        axes[row_idx, col_idx].axis("off")
        # Visualize 2D hand joints
        if pred_handjoints2d is not None:
            visualize_joints_2d(axes[row_idx, col_idx], pred_handjoints2d[row_idx], alpha=1, joint_idxs=False)
        if gt_handjoints2d is not None:
            visualize_joints_2d(axes[row_idx, col_idx], gt_handjoints2d[row_idx], alpha=0.5, joint_idxs=False)

        # Column 1
        col_idx = 1
        axes[row_idx, col_idx].imshow(images[row_idx])
        axes[row_idx, col_idx].axis("off")
        # axes[row_idx, col_idx].set_title("dvec: {:.2f},{:.2f},{:.2f}".format(tmp[row_idx, 0, 0], tmp[row_idx, 0, 1], tmp[row_idx, 0, 2]))
        if gt_objfps2d is not None and pred_objfps2d is not None:
            arrow_nb = gt_objfps2d.shape[1]
            idxs = range(arrow_nb)
            arrows = torch.cat([gt_objfps2d[:, idxs].float(), pred_objfps2d[:, idxs].float()], 1)
            links = [[i, i + arrow_nb] for i in idxs]
            visualize_joints_2d(
                axes[row_idx, col_idx],
                arrows[row_idx],
                alpha=0.5,
                joint_idxs=False,
                links=links,
                color=["k"] * arrow_nb,
            )
        if pred_objvar2d is not None:
            ells = compute_confidence_ellipses(pred_objfps2d[row_idx], pred_objvar2d[row_idx])
            for e in ells:
                axes[row_idx, col_idx].add_artist(e)
        if pred_objfps2d is not None:
            axes[row_idx, col_idx].scatter(
                pred_objfps2d[row_idx, :, 0], pred_objfps2d[row_idx, :, 1], c="r", s=2, marker="X", alpha=0.7
            )
        if gt_objfps2d is not None:
            axes[row_idx, col_idx].scatter(
                gt_objfps2d[row_idx, :, 0], gt_objfps2d[row_idx, :, 1], c="b", s=2, marker="X", alpha=0.7
            )
        # if gt_objfps2d is not None:
        #     axes[row_idx, col_idx].scatter(
        #         gt_objfps2d[row_idx, [0,5], 0], gt_objfps2d[row_idx, [0,5], 1], c="g", s=4, marker="o", alpha=0.7
        #     )


        # Column 2
        col_idx = 2
        axes[row_idx, col_idx].imshow(images[row_idx])
        axes[row_idx, col_idx].axis("off")
        # Visualize 2D object vertices
        if pred_objverts2d is not None:
            axes[row_idx, col_idx].scatter(
                pred_objverts2d[row_idx, :, 0], pred_objverts2d[row_idx, :, 1], c="r", s=1, alpha=0.1
            )
        if gt_objverts2d is not None:
            axes[row_idx, col_idx].scatter(
                gt_objverts2d[row_idx, :, 0], gt_objverts2d[row_idx, :, 1], c="b", s=1, alpha=0.02
            )
        # Visualize 2D object bounding box
        # if pred_objcorners2d is not None:
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         pred_objcorners2d[row_idx],
        #         alpha=1,
        #         joint_idxs=False,
        #         links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
        #     )
        # if gt_objcorners2d is not None:
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         gt_objcorners2d[row_idx],
        #         alpha=0.5,
        #         joint_idxs=False,
        #         links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
        #     )
        # Visualize some (vertex position) errors for the 2D object vertices
        if (gt_objfps2d is None or pred_objfps2d is None) and gt_objverts2d is not None and pred_objverts2d is not None:
            idxs = list(range(6))
            arrow_nb = len(idxs)
            arrows = torch.cat([gt_objverts2d[:, idxs].float(), pred_objverts2d[:, idxs].float()], 1)
            links = [[i, i + arrow_nb] for i in range(arrow_nb)]
            visualize_joints_2d(
                axes[row_idx, col_idx],
                arrows[row_idx],
                alpha=0.5,
                joint_idxs=False,
                links=links,
                color=["k"] * arrow_nb,
            )


        # Column 3
        # view from the top
        col_idx = 3
        #axes[row_idx, col_idx].set_title("rotY: {:.1f}".format(gt_drill_angle_Y[row_idx]))
        if gt_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                gt_objverts3dw[row_idx, :, 2], gt_objverts3dw[row_idx, :, 0], c="b", s=1, alpha=0.02
            )
        if pred_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                pred_objverts3dw[row_idx, :, 2], pred_objverts3dw[row_idx, :, 0], c="r", s=1, alpha=0.02
            )

        if pred_handjoints3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx], pred_handjoints3dw[row_idx, :, [2,0]], alpha=1, joint_idxs=False
            )
        if gt_handjoints3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx], gt_handjoints3dw[row_idx, :, [2,0]], alpha=0.5, joint_idxs=False
            )
        axes[row_idx, col_idx].invert_yaxis()

        # if pred_objcorners3dw is not None:
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         pred_objcorners3dw[row_idx],
        #         alpha=1,
        #         joint_idxs=False,
        #         links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
        #     )
        # if gt_objcorners3dw is not None:
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         gt_objcorners3dw[row_idx],
        #         alpha=0.5,
        #         joint_idxs=False,
        #         links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
        #     )
        # if pred_objverts3dw is not None and gt_objverts3dw is not None:
        #     arrow_nb = 6
        #     arrows = torch.cat([gt_objverts3dw[:, :arrow_nb], pred_objverts3dw[:, :arrow_nb]], 1)
        #     links = [[i, i + arrow_nb] for i in range(arrow_nb)]
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         arrows[row_idx],
        #         alpha=0.5,
        #         joint_idxs=False,
        #         links=links,
        #         color=["k"] * arrow_nb,
        #     )

        # Column 4
        # view from the right
        col_idx = 4
        #axes[row_idx, col_idx].set_title("rotX: {:.1f}".format(gt_drill_angle_X[row_idx]))
        # invert second axis here for more consistent viewpoints
        if gt_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                gt_objverts3dw[row_idx, :, 2], -gt_objverts3dw[row_idx, :, 1], c="b", s=1, alpha=0.02
            )
        if pred_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                pred_objverts3dw[row_idx, :, 2], -pred_objverts3dw[row_idx, :, 1], c="r", s=1, alpha=0.02
            )
        if pred_handjoints3dw is not None:
            pred_handjoints3dw_inv = np.stack([pred_handjoints3dw[:, :, 2], -pred_handjoints3dw[:, :, 1]], axis=-1)
            visualize_joints_2d(
                axes[row_idx, col_idx], pred_handjoints3dw_inv[row_idx, :, :], alpha=1, joint_idxs=False
            )
        if gt_handjoints3dw is not None:
            gt_handjoints3dw_inv = np.stack([gt_handjoints3dw[:, :, 2], -gt_handjoints3dw[:, :, 1]], axis=-1)
            visualize_joints_2d(
                axes[row_idx, col_idx], gt_handjoints3dw_inv[row_idx, :, :], alpha=0.5, joint_idxs=False
            )
        # if pred_objcorners3dw is not None:
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         pred_objcorners3dw[row_idx, :, 1:],
        #         alpha=1,
        #         joint_idxs=False,
        #         links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
        #     )
        # if gt_objcorners3dw is not None:
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         gt_objcorners3dw[row_idx, :, 1:],
        #         alpha=0.5,
        #         joint_idxs=False,
        #         links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
        #     )
        # if pred_objverts3dw is not None and gt_objverts3dw is not None:
        #     arrow_nb = 6
        #     arrows = torch.cat([gt_objverts3dw[:, :arrow_nb, 1:], pred_objverts3dw[:, :arrow_nb, 1:]], 1)
        #     links = [[i, i + arrow_nb] for i in range(arrow_nb)]
        #     visualize_joints_2d(
        #         axes[row_idx, col_idx],
        #         arrows[row_idx],
        #         alpha=0.5,
        #         joint_idxs=False,
        #         links=links,
        #         color=["k"] * arrow_nb,
        #     )

        if display_centered:
            # Column 5
            col_idx = 5
            if gt_canobjverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_canobjverts3d[row_idx, :, 0], gt_canobjverts3d[row_idx, :, 1], c="b", s=1, alpha=0.02
                )
            if pred_objverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    pred_objverts3d[row_idx, :, 0], pred_objverts3d[row_idx, :, 1], c="r", s=1, alpha=0.02
                )
            if pred_objcorners3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx],
                    pred_objcorners3d[row_idx],
                    alpha=1,
                    joint_idxs=False,
                    links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
                )
            if gt_canobjcorners3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx],
                    gt_canobjcorners3d[row_idx],
                    alpha=0.5,
                    joint_idxs=False,
                    links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
                )
            if pred_objcorners3d is not None and gt_canobjcorners3d is not None:
                arrow_nb = 6
                arrows = torch.cat([gt_canobjcorners3d[:, :arrow_nb], pred_objcorners3d[:, :arrow_nb]], 1)
                links = [[i, i + arrow_nb] for i in range(arrow_nb)]
                visualize_joints_2d(
                    axes[row_idx, col_idx],
                    arrows[row_idx],
                    alpha=0.5,
                    joint_idxs=False,
                    links=links,
                    color=["k"] * arrow_nb,
                )
            axes[row_idx, col_idx].set_aspect("equal")
            axes[row_idx, col_idx].invert_yaxis()

            # Column 6
            col_idx = 6
            if gt_objverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_objverts3d[row_idx, :, 0], gt_objverts3d[row_idx, :, 1], c="b", s=1, alpha=0.02
                )
            # if pred_objverts3d is not None:
            #     axes[row_idx, 2].scatter(
            #         pred_objverts3d[row_idx, :, 0], pred_objverts3d[row_idx, :, 1], c="r", s=1, alpha=0.02
            #     )
            if gt_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_handverts3d[row_idx, :, 0], gt_handverts3d[row_idx, :, 1], c="g", s=1, alpha=0.2
                )
            if pred_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    pred_handverts3d[row_idx, :, 0], pred_handverts3d[row_idx, :, 1], c="c", s=1, alpha=0.2
                )
            if pred_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], pred_handjoints3d[row_idx], alpha=1, joint_idxs=False
                )
            if gt_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], gt_handjoints3d[row_idx], alpha=0.5, joint_idxs=False
                )
            axes[row_idx, col_idx].invert_yaxis()

            # Column 7
            col_idx = 7
            if gt_objverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_objverts3d[row_idx, :, 1], gt_objverts3d[row_idx, :, 2], c="b", s=1, alpha=0.02
                )
            # if pred_objverts3d is not None:
            #     axes[row_idx, 3].scatter(
            #         pred_objverts3d[row_idx, :, 1], pred_objverts3d[row_idx, :, 2], c="r", s=1, alpha=0.02
            #     )
            if gt_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_handverts3d[row_idx, :, 1], gt_handverts3d[row_idx, :, 2], c="g", s=1, alpha=0.2
                )
            if pred_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    pred_handverts3d[row_idx, :, 1], pred_handverts3d[row_idx, :, 2], c="c", s=1, alpha=0.2
                )
            if pred_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], pred_handjoints3d[row_idx][:, 1:], alpha=1, joint_idxs=False
                )
            if gt_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], gt_handjoints3d[row_idx][:, 1:], alpha=0.5, joint_idxs=False
                )

    consistdisplay.squashfig(fig)
    fig.savefig(save_img_path, dpi=300)
