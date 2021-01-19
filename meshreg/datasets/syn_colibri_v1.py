import os
import pickle
import random
from glob import glob
import re
import pickle
import trimesh

import numpy as np
from PIL import Image, ImageFile
import torch
from manopth import manolayer

from libyana.transformutils import handutils
from libyana.meshutils import meshnorm

from meshreg.datasets import syn_colibri_v1_utils as utils
from meshreg.datasets.queries import BaseQueries, get_trans_queries

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SynColibriV1(object):

    def __init__(
        self,
        root="data/syn_colibri_v1",
        split="train",
        joint_nb=21,
        use_cache=True
    ):
        """
        Args:
            # TODO
        """
        super().__init__()
        self.name = "syn_colibri_v1"
        self.root = root
        self.split = split
        self.joint_nb = joint_nb
        self.has_dist2strong = False
        self.use_cache = use_cache
        self.rgb_root = os.path.join(self.root, "rgb")
        self.segm_root = os.path.join(self.root, "segm")
        self.meta_root = os.path.join(self.root, "meta")
        self.cache_folder = os.path.join("data", "cache", "syn_colibri_v1")
        os.makedirs(self.cache_folder, exist_ok=True)

        # Get queries
        self.all_queries = [
            BaseQueries.IMAGE,
            BaseQueries.SIDE,
            BaseQueries.CAMINTR,
            BaseQueries.JOINTS2D,
            BaseQueries.JOINTS3D,
            BaseQueries.JOINTVIS,
            BaseQueries.HANDVERTS2D,
            BaseQueries.HANDVERTS3D,
            BaseQueries.OBJCORNERS3D,
            BaseQueries.OBJFPS2D,
            BaseQueries.OBJFPS3D,
            BaseQueries.OBJFPSVECFIELD,
            BaseQueries.OBJVERTS2D,
            BaseQueries.OBJVERTS3D,
            BaseQueries.OBJCANVERTS,
            BaseQueries.OBJMASK,
            BaseQueries.OBJFACES,
            BaseQueries.OBJPOSE,
        ]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        self.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root="assets/mano",
            center_idx=None,
            flat_hand_mean=True,
        )

        self.obj_meshes = {}
        self.obj_fps_3d = {}
        self.load_dataset()

        # Infor for rendering
        self.image_size = np.array([256, 256])
        print("Got {} samples for split {}".format(len(self.samples), self.split))

        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def load_dataset(self):
        cache_path = os.path.join(self.cache_folder, f"{self.split}.pkl")
        fps_path = os.path.join(self.root, f"fps.txt")
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, "rb") as cache_f:
                self.samples = pickle.load(cache_f)
            print("Cached information for dataset {} loaded from {}".format(self.name, cache_path))
        else:
            self.samples = []
            # Load meta data
            split_file = os.path.join(self.root, f"{self.split}.txt")
            if os.path.exists(split_file):
                sample_names = []
                with open(split_file, "r") as f:
                    sample_names = f.read().splitlines()
                print(len(sample_names))

                for sample_name in sample_names:
                    parsed = re.search("(.+)_grasp([0-9]+)_([0-9]+)", sample_name)
                    sample_file = os.path.join(self.meta_root, f"{sample_name}.pkl")
                    sample = pickle.load(open(sample_file, "rb"))
                    sample['model_name'] = parsed.group(1)
                    sample['grasp_no'] = parsed.group(2)
                    sample['frame_no'] = parsed.group(3)
                    self.samples.append(sample)

    def get_dataidx(self, idx):
        assert idx < len(self), "Index out of range!"
        return self.samples[idx]

    def get_invdataidx(self, sample):
        return self.samples.index(sample)

    def get_image(self, idx):
        sample = self.get_dataidx(idx)
        img_path = os.path.join(self.rgb_root, "{}_grasp{}_{}.jpg".format(sample['model_name'], sample['grasp_no'], sample['frame_no']))
        img = Image.open(img_path).convert("RGB")
        return img

    def get_obj_mask(self, idx):
        sample = self.get_dataidx(idx)
        img_path = os.path.join(self.segm_root, "{}_grasp{}_{}.png".format(sample['model_name'], sample['grasp_no'],
                                                                          sample['frame_no']))
        img = Image.open(img_path).convert("RGB")
        obj_mask = img.getchannel('R')
        return (np.array(obj_mask) != 0).astype(np.uint8)

    def get_hand_verts3d(self, idx):
        sample = self.get_dataidx(idx)
        verts3d = sample['verts_3d']
        hom_verts3d = np.concatenate([verts3d, np.ones((verts3d.shape[0], 1))], 1).transpose() # shape: (4,778)
        trans_verts3d = np.dot(self.get_camextr(idx), hom_verts3d).transpose() # shape: (778,3)
        return np.array(trans_verts3d).astype(np.float32)

    def get_hand_verts2d(self, idx):
        verts3d = self.get_hand_verts3d(idx) # shape: (778,3)
        assert verts3d.shape[1] == 3, "get_hand_verts3d returned wrong shape: {}".format(verts3d.shape)
        hom_verts2d = np.dot(self.get_camintr(idx), verts3d.transpose())  # shape: (3,778)
        # Normalize homogeneous coordinates
        hom_verts2d = hom_verts2d / hom_verts2d[2, :]
        verts2d = hom_verts2d[:2, :].transpose() # shape: (778,2)
        return np.array(verts2d).astype(np.float32)

    def get_jointvis(self, idx):
        return np.ones(self.joint_nb)

    def get_obj_mesh(self, idx):
        sample = self.get_dataidx(idx)
        if not sample['obj_path'] in self.obj_meshes:
            mesh = trimesh.load(os.path.join(self.root, sample['obj_path']))
            self.obj_meshes[sample['obj_path']] = mesh
        return self.obj_meshes[sample['obj_path']]

    def get_obj_faces(self, idx):
        faces = self.get_obj_mesh(idx).faces
        return np.array(faces).astype(np.float32)

    def get_obj_verts(self, idx):
        verts = self.get_obj_mesh(idx).vertices
        return np.array(verts).astype(np.float32)

    def get_obj_pose(self, idx):
        # Get object pose (3,4) matrix in world coordinate frame
        sample = self.get_dataidx(idx)
        transform = self.get_camextr(idx) @ sample['affine_transform']
        return np.array(transform).astype(np.float32)

    def get_obj_verts_trans(self, idx):
        # Get object 3d vertices (n,3) in the camera coordinate frame
        verts = self.get_obj_verts(idx) # shape: (n, 3)
        trans_verts = utils.transform(verts, self.get_obj_pose(idx), convert_to_homogeneous=True)
        return np.array(trans_verts).astype(np.float32)

    def get_obj_verts_can(self, idx, no_center=False):
        verts = self.get_obj_verts(idx) # shape: (n, 3)
        if no_center:
            return verts, None, None
        else:
            return meshnorm.center_vert_bbox(verts, scale=False)

    def get_objverts2d(self, idx):
        objpoints3d = self.get_obj_verts_trans(idx).transpose() # shape: (3, n)
        hom_2d = np.dot(self.get_camintr(idx), objpoints3d).transpose() # shape: (n, 3)
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2] # shape: (n, 2)
        return np.array(verts2d).astype(np.float32)


    def get_obj_fps3d(self, idx):
        # FPS3D HAVE TO BE IN THE OBJECT COORDINATE SYSTEM (SINCE WE'D LIKE TO RECOVER THE POSE VIA PNP)!
        sample = self.get_dataidx(idx)
        if sample['model_name'] not in self.obj_fps_3d:
            fps_path = os.path.join(self.root, "{}.txt".format(sample['model_name']))
            fps3d = np.loadtxt(fps_path) # (8,3) in object coordinate frame
            # Get 3d center in object coordinates
            model = self.get_obj_verts(idx)
            min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
            min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
            min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
            center3d = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0]
            fps3d = np.concatenate([fps3d, np.expand_dims(center3d, 0)])
            self.obj_fps_3d[sample['model_name']] = fps3d

        fps3d = np.array(self.obj_fps_3d[sample['model_name']])
        return np.array(fps3d).astype(np.float32)


    def get_obj_fps3d_trans(self, idx):
        fps3d = self.get_obj_fps3d(idx)
        pose = self.get_obj_pose(idx)
        # Transform points from tool coordinate frame to camera coordinate frame
        fps3d = utils.transform(fps3d, pose, convert_to_homogeneous=True)
        return np.array(fps3d).astype(np.float32)

    def get_obj_fps2d(self, idx):
        # FPS2D ARE IN CAMERA COORDINATES (IN ORDER TO EVALUATE THE ACCURACY OF THE 2D KEYPOINT ESTIMATES)!
        fps3d = self.get_obj_fps3d_trans(idx)
        intr = self.get_camintr(idx)
        fps2d_hom = utils.transform(fps3d, intr)
        fps2d = fps2d_hom[:, :2] / fps2d_hom[:, 2:]
        return np.array(fps2d).astype(np.float32)

    def get_obj_fpsvectorfield(self, idx):
        obj_mask = self.get_obj_mask(idx)
        fps2d = self.get_obj_fps2d(idx)
        vector_field = utils.compute_vertex(obj_mask, fps2d).transpose(2, 0, 1)
        return vector_field

    def get_obj_corners3d(self, idx):
        model = self.get_obj_verts_trans(idx)
        min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
        min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
        min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return np.array(corners_3d).astype(np.float32)

    def get_objcorners2d(self, idx):
        corners_3d = self.get_obj_corners3d(idx)
        intr = self.get_camintr(idx)
        corners_2d_hom = utils.transform(corners_3d, intr)
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:]
        return np.array(corners_2d).astype(np.float32)

    def get_obj_center3d(self, idx):
        corners_3d = self.get_obj_corners3d(idx)
        center_3d = (np.max(corners_3d, 0) + np.min(corners_3d, 0)) / 2
        return np.array(center_3d).astype(np.float32)

    def get_joints3d(self, idx):
        sample = self.get_dataidx(idx)
        coords3d = sample['coords_3d'] # shape: (n, 3)
        trans_coords3d = utils.transform(coords3d, self.get_camextr(idx), convert_to_homogeneous=True)
        return np.array(trans_coords3d).astype(np.float32)

    def get_joints2d(self, idx):
        sample = self.get_dataidx(idx)
        return np.array(sample['coords_2d']).astype(np.float32)

    def get_camintr(self, idx):
        sample = self.get_dataidx(idx)
        return np.array(sample['cam_calib']).astype(np.float32) # shape: (3, 3)

    def get_camextr(self, idx):
        sample = self.get_dataidx(idx)
        return np.array(sample['cam_extr']).astype(np.float32) # shape: (3, 4)

    def get_sides(self, idx):
        sample = self.get_dataidx(idx)
        return sample['side']

    def get_meta(self, idx):
        sample = self.get_dataidx(idx)
        meta = {"objname": sample['model_name']}
        return meta

    def get_center_scale(self, idx):
        center = (self.image_size / 2).astype(np.int)
        scale = self.image_size.max()
        return center, scale

    def get_objvis2d(self, idx):
        objvis = np.ones_like(self.get_objverts2d(idx))
        return objvis

    def get_handvis2d(self, idx):
        handvis = np.ones_like(self.get_handverts2d(idx))
        return handvis

    def __len__(self):
        return len(self.samples)
