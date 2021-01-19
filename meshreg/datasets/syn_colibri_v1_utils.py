from functools import lru_cache
import os
import pickle
import numpy as np

@lru_cache(128)
def load_manoinfo(pkl_path):
    with open(pkl_path, "rb") as p_f:
        data = pickle.load(p_f)
    return data


def transform(verts, trans, convert_to_homogeneous=False):
    assert len(verts.shape) == 2, "Expected 2 dimensions for verts, got: {}.".format(len(verts.shape))
    assert len(trans.shape) == 2, "Expected 2 dimensions for trans, got: {}.".format(len(trans.shape))
    if convert_to_homogeneous:
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    else:
        hom_verts = verts

    assert trans.shape[1] == hom_verts.shape[1], \
        "Incompatible shapes: verts.shape: {}, trans.shape: {}".format(verts.shape, trans.shape)

    trans_verts = np.dot(trans, hom_verts.transpose()).transpose()
    return trans_verts


def compute_vertex(mask, kpt_2d):
    h, w = mask.shape
    m = kpt_2d.shape[0]
    xy = np.argwhere(mask != 0)[:, [1, 0]]

    vertex = kpt_2d[None] - xy[:, None]
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm < 1e-3] += 1e-3
    vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    vertex_out = np.reshape(vertex_out, [h, w, m * 2])

    return vertex_out