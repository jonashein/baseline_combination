import warnings
import torch
import numpy as np
from meshreg.datasets.queries import TransQueries, BaseQueries

class Metric:

    def __init__(self, name, unit, axis_label):
        self._data = {}
        pass

    def get_name(self):
        pass

    def get_axis_label(self):
        pass

    def get_axis_limits(self):
        pass

    def get_unit(self):
        pass

    def eval(self, epoch, split, sample, results):
        pass


def _recover_back(joints_trans, affinetrans):
    """
    Given 2d point coordinates and an affine transform, recovers original pixel points
    (locations before translation, rotation, crop, scaling... are applied during data
    augmentation)
    """
    batch_size = joints_trans.shape[0]
    point_nb = joints_trans.shape[1]
    hom2d = torch.cat([joints_trans, joints_trans.new_ones(batch_size, point_nb, 1)], -1)
    rec2d = torch.inverse(affinetrans).bmm(hom2d.transpose(1, 2).float()).transpose(1, 2)[:, :, :2]
    return rec2d


def _recover_3d_proj(joints3d, joints2d, camintr, est_scale, est_trans, center_idx=9):
    # Estimate scale and trans between 3D and 2D
    trans3d = joints3d[:, center_idx : center_idx + 1]
    joints3d_c = joints3d - trans3d
    focal = camintr[:, :1, :1]
    est_Z0 = focal / est_scale
    est_XY0 = (est_trans[:, 0] - camintr[:, :2, 2]) * est_Z0[:, 0] / focal[:, 0]
    est_c3d = torch.cat([est_XY0, est_Z0[:, 0]], -1).unsqueeze(1)
    recons3d = est_c3d + joints3d_c
    return recons3d, est_c3d


def _transform(points3d, Rt):
    # points3d: (B,N,3)
    # Rt: (B,3,4)
    hom_points3d = np.concatenate([points3d, np.ones([points3d.shape[0], points3d.shape[1], 1])], axis=2)
    trans_points3d = hom_points3d @ Rt.transpose((0,2,1))
    return trans_points3d

def _euclidean_dist(gt, pred, compute_mean_of_keypoints=True):
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    #gt = np.squeeze(gt)
    #pred = np.squeeze(pred)

    assert gt.ndim == 3, "gt not 3-dim, but has shape {}".format(gt.shape)
    assert pred.ndim == 3, "pred not 3-dim, but has shape {}".format(pred.shape)
    # shapes: (batch_size, nb_keypoints, point_dim), e.g. (64, 21, 3)

    # calc euclidean distance
    euclidean_dist = np.linalg.norm(gt - pred, ord=2, axis=-1)
    if compute_mean_of_keypoints:
        euclidean_dist = np.mean(euclidean_dist, axis=-1)
    return euclidean_dist


def hand_joints_2d(batch, pred):
    result = {}
    if "joints2d" in pred and BaseQueries.JOINTS2D in batch:
        gt_joints2d = batch[TransQueries.JOINTS2D]
        affinetrans = batch[TransQueries.AFFINETRANS]
        or_joints2d = batch[BaseQueries.JOINTS2D]
        rec_pred = _recover_back(pred["joints2d"].detach().cpu(), affinetrans)
        rec_gt = _recover_back(gt_joints2d, affinetrans)
        # Sanity check, this should be ~1pixel
        gt_err = np.linalg.norm(rec_gt - or_joints2d, 2, -1).mean()
        if gt_err > 1:
            warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
        result["hand_joints_2d"] = _euclidean_dist(or_joints2d.detach().cpu().numpy(), rec_pred.numpy())
    return result


def hand_joints_3d(batch, pred):
    result = {}
    if "recov_joints3d" in pred and TransQueries.JOINTS3D in batch and TransQueries.CENTER3D in batch:
        gt_joints3d = batch[TransQueries.JOINTS3D] + np.expand_dims(batch[TransQueries.CENTER3D], axis=1)
        pred_joints3d = pred["recov_joints3d"].detach().cpu()
        result["hand_joints_3d"] = _euclidean_dist(gt_joints3d, pred_joints3d)
    return result


def hand_joints_3d_cent(batch, pred, idxs=None, center_idx=9):
    result = {}
    if "joints3d" in pred and TransQueries.JOINTS3D in batch:
        gt_joints3d = batch[TransQueries.JOINTS3D]
        pred_joints3d = pred["joints3d"].detach().cpu()
        if center_idx is not None:
            gt_joints3d_cent = gt_joints3d - gt_joints3d[:, center_idx : center_idx + 1]
            pred_joints3d_cent = pred_joints3d - pred_joints3d[:, center_idx : center_idx + 1]
            result["hand_joints_3d_cent"] = _euclidean_dist(gt_joints3d_cent, pred_joints3d_cent)
    return result


def obj_corners_2d(batch, pred):
    result = {}
    if("obj_corners2d" in pred and pred["obj_corners2d"] is not None and BaseQueries.OBJCORNERS2D in batch):
        obj_corners2d_gt = batch[TransQueries.OBJCORNERS2D]
        affinetrans = batch[TransQueries.AFFINETRANS]
        or_corners2d = batch[BaseQueries.OBJCORNERS2D]
        rec_pred = recover_back(pred["obj_corners2d"].detach().cpu(), affinetrans)
        rec_gt = recover_back(obj_corners2d_gt, affinetrans)
        # Sanity check, this should be ~1pixel
        gt_err = np.linalg.norm(rec_gt - or_corners2d, 2, -1).mean()
        if gt_err > 1:
            warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
        result["corners2d_base"] = _euclidean_dist(or_corners2d.detach().cpu().numpy(), rec_pred.numpy())
    return result


def obj_verts_2d(batch, pred):
    result = {}
    if "obj_verts2d" in pred and pred["obj_verts2d"] is not None and BaseQueries.OBJVERTS2D in batch:
        obj_verts2d_gt = batch[TransQueries.OBJVERTS2D]
        affinetrans = batch[TransQueries.AFFINETRANS]
        or_verts2d = batch[BaseQueries.OBJVERTS2D]
        obj_verts2d_pred = pred["obj_verts2d"]
        rec_pred = _recover_back(obj_verts2d_pred, affinetrans)
        rec_gt = _recover_back(obj_verts2d_gt, affinetrans)
        # Sanity check, this should be ~1pixel
        gt_err = np.linalg.norm(rec_gt - or_verts2d, 2, -1).mean()
        if gt_err > 1:
            warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
        result["obj_verts_2d"] = _euclidean_dist(rec_pred, or_verts2d.cpu())
    return result


def obj_verts_3d(batch, pred):
    result = {}
    if "recov_objverts3d" in pred and pred["recov_objverts3d"] is not None and BaseQueries.OBJVERTS3D in batch:
        or_verts3d = batch[BaseQueries.OBJVERTS3D]
        verts_3d_pred = pred["recov_objverts3d"]
        if isinstance(verts_3d_pred, torch.Tensor):
            verts_3d_pred.cpu()
        result["obj_verts_3d"] = _euclidean_dist(verts_3d_pred, or_verts3d)
    return result


def obj_pose_3d(batch, pred):
    result = {}
    if "obj_pose" in pred and pred["obj_pose"] is not None and BaseQueries.OBJPOSE in batch:
        pose_pred = pred["obj_pose"]
        if isinstance(pose_pred, torch.Tensor):
            pose_pred = pose_pred.detach().cpu().numpy()

        pose_targets = batch[BaseQueries.OBJPOSE]
        if isinstance(pose_targets, torch.Tensor):
            pose_targets = pose_targets.numpy()

        # Apply object vertex normalization onto target pose (pose_pred is the pose for the normalized object)
        normalization = np.tile(np.eye(4), (pose_targets.shape[0], 1, 1))
        if BaseQueries.OBJCANTRANS in batch and batch[BaseQueries.OBJCANTRANS] is not None:
            objcantrans = batch[BaseQueries.OBJCANTRANS]
            if isinstance(objcantrans, torch.Tensor):
                objcantrans = objcantrans.detach().cpu().numpy()
            normalization[:, :3, 3] = objcantrans
        if BaseQueries.OBJCANSCALE in batch and batch[BaseQueries.OBJCANSCALE] is not None:
            objcanscale = batch[BaseQueries.OBJCANSCALE]
            if isinstance(objcanscale, torch.Tensor):
                objcanscale = objcanscale.detach().cpu().numpy()
            objcanscale = objcanscale.reshape((-1, 1, 1))
            normalization = np.multiply(normalization, np.tile(objcanscale, (1, 4, 4)))
        pose_targets = np.matmul(pose_targets, normalization)

        # Compute 3d translational error
        result["obj_translation_3d"] = np.linalg.norm(pose_pred[:, :3, 3] - pose_targets[:, :3, 3], axis=1)
        result["obj_depth_3d"] = np.abs(pose_pred[:, 2, 3] - pose_targets[:, 2, 3])

        # Compute rotational error
        rot_pred = pose_pred[:, :3, :3]
        rot_target_T = pose_targets[:, :3, :3].transpose((0, 2, 1))
        rotation_diff = np.matmul(rot_pred, rot_target_T)
        trace = np.trace(rotation_diff, axis1=1, axis2=2)
        # Check for singularities at 0° or 180°
        is_symmetric = np.all(np.isclose(rot_pred, rot_pred.transpose((0, 2, 1))), axis=(1, 2))
        batched_identity = np.tile(np.eye(3), (rot_pred.shape[0], 1, 1))
        is_identity = np.all(np.isclose(rot_pred, batched_identity), axis=(1, 2))
        # Overwrite trace=-1.0 if we're at the singularity at 180°
        trace = np.where(is_symmetric, -1.0, trace)
        # Overwrite trace=3.0 if we're at the singularity at 0°
        trace = np.where(is_identity, 3.0, trace)
        #trace = np.where(np.logical_and(trace > 3.0, trace < 3.1), 3.0, trace)
        #trace = np.where(np.logical_and(trace > -1.1, trace < -1.0), -1.0, trace)
        obj_rotation_3d = np.rad2deg(np.arccos((trace - 1.) / 2.))
        result["obj_rotation_3d"] = obj_rotation_3d

        #if np.any(np.isnan(obj_rotation_3d)) or np.any(obj_rotation_3d > 180.0):
        #    print("obj_rotation_3d:\n{}".format(obj_rotation_3d))
        #    print("rot_pred:\n{}".format(rot_pred))
        #    print("rot_target_T:\n{}".format(rot_target_T))
        #    print("rotation_diff:\n{}".format(rotation_diff))
        #    print("trace:\n{}".format(trace))

    return result


def obj_drilltip_3d(batch, pred):
    result = {}
    if "obj_pose" in pred and pred["obj_pose"] is not None and BaseQueries.OBJPOSE in batch:
        pose_pred = pred["obj_pose"]
        if isinstance(pose_pred, torch.Tensor):
            pose_pred = pose_pred.detach().cpu().numpy()

        pose_targets = batch[BaseQueries.OBJPOSE]
        if isinstance(pose_targets, torch.Tensor):
            pose_targets = pose_targets.numpy()

        # Apply object vertex normalization onto target pose (pose_pred is the pose for the normalized object)
        normalization = np.tile(np.eye(4), (pose_targets.shape[0], 1, 1))
        if BaseQueries.OBJCANTRANS in batch and batch[BaseQueries.OBJCANTRANS] is not None:
            objcantrans = batch[BaseQueries.OBJCANTRANS]
            if isinstance(objcantrans, torch.Tensor):
                objcantrans = objcantrans.detach().cpu().numpy()
            normalization[:, :3, 3] = objcantrans
        if BaseQueries.OBJCANSCALE in batch and batch[BaseQueries.OBJCANSCALE] is not None:
            objcanscale = batch[BaseQueries.OBJCANSCALE]
            if isinstance(objcanscale, torch.Tensor):
                objcanscale = objcanscale.detach().cpu().numpy()
            objcanscale = objcanscale.reshape((-1, 1, 1))
            normalization = np.multiply(normalization, np.tile(objcanscale, (1, 4, 4)))
        pose_targets = np.matmul(pose_targets, normalization)

        # Compute drill bit position, orientation
        # THIS IS ONLY VALID FOR OUR EXACT DRILL MODEL!
        DRILL_TIP = np.array([0.053554, 0.225361, -0.241646])
        DRILL_SHANK = np.array([0.057141, 0.220794, -0.121545])
        batched_drill_tip = np.tile(np.expand_dims(DRILL_TIP, axis=(0,1)), (pose_pred.shape[0], 1, 1))
        batched_drill_shank = np.tile(np.expand_dims(DRILL_SHANK, axis=(0, 1)), (pose_pred.shape[0], 1, 1))

        pred_drill_tip = _transform(batched_drill_tip, pose_pred)
        pred_drill_shank = _transform(batched_drill_shank, pose_pred)
        gt_drill_tip = _transform(batched_drill_tip, pose_targets)
        gt_drill_shank = _transform(batched_drill_shank, pose_targets)
        result["obj_drilltip_trans_3d"] = _euclidean_dist(gt_drill_tip, pred_drill_tip)

        pred_drill_vec = pred_drill_tip - pred_drill_shank
        pred_drill_vec = pred_drill_vec / np.expand_dims(np.linalg.norm(pred_drill_vec, axis=2), axis=1)
        gt_drill_vec = gt_drill_tip - gt_drill_shank
        gt_drill_vec = gt_drill_vec / np.expand_dims(np.linalg.norm(gt_drill_vec, axis=2), axis=1)
        dotprod = pred_drill_vec @ gt_drill_vec.transpose((0, 2, 1))
        result["obj_drilltip_rot_3d"] = np.rad2deg(np.arccos(dotprod.squeeze()))
        result["gt_drill_rot"] = np.squeeze(gt_drill_vec, axis=1)
    return result


def obj_kpt_2d(batch, pred):
    result = {}

    if BaseQueries.OBJFPS2D in batch and "kpt_2d" in pred and pred["kpt_2d"] is not None:
        gt_fps2d = batch[BaseQueries.OBJFPS2D]
        affinetrans = batch[TransQueries.AFFINETRANS]
        rec_pred = _recover_back(pred["kpt_2d"].detach().cpu(), affinetrans)
        rec_gt = _recover_back(gt_fps2d, affinetrans)
        # Sanity check, this should be ~1pixel
        gt_err = np.linalg.norm(rec_gt - gt_fps2d, 2, -1).mean()
        if gt_err > 1:
            warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
        gt_fps2d = gt_fps2d.detach().cpu().numpy()
        result["obj_keypoints_2d"] = _euclidean_dist(gt_fps2d, rec_pred.numpy())

        # result["drill_orientation_2d"] = gt_fps2d[:, 5, ] - gt_fps2d[:, 0, ]

    return result


def evaluate(batch, pred):
    metrics = {
        hand_joints_2d,
        hand_joints_3d,
        hand_joints_3d_cent,
        obj_corners_2d,
        obj_verts_2d,
        obj_verts_3d,
        obj_pose_3d,
        obj_kpt_2d,
        obj_drilltip_3d,
    }

    eval = {}
    for func in metrics:
        pred = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in pred.items()}
        result = func(batch, pred)
        eval.update(result)
    return eval