import argparse
import os
import numpy as np
from meshreg.netscripts.monitor import MetricMonitor

def main(args):
    assert os.path.isfile(args.metrics), f"Invalid argument for metrics: {args.metrics}"

    monitor = MetricMonitor(filepath=args.metrics)
    # merge all batch metrics into single array
    for m, mdict in monitor.data.items():
         for s, sdict in mdict.items():
             tmp = []
             for e, value in sdict.items():
                 tmp += [v.reshape((-1, 1)) for v in value]
             sdict.clear()
             monitor.data[m][s][0] = np.concatenate(tmp)

    monitor = remove_outliers(monitor)

    # Nicer output of relevant metrics
    metrics = ["obj_verts_3d", "obj_verts_2d", "obj_drilltip_trans_3d", "obj_drilltip_rot_3d", "obj_keypoints_2d", "hand_joints_3d", "hand_joints_2d"]
    metric_names = ["Tool ADD (mm)", "Tool Proj2D (px)", "Drill tip error (mm)", "Drill bit direction error (deg)", "2D Keypoint error (px)", "Hand ADD (mm)", "Hand Proj2D (px)"]
    metric_scales = [1000.0, 1.0, 1000.0, 1.0, 1.0, 1000.0, 1.0]
    means = monitor.means_per_epoch(metrics=metrics)
    for i, m in enumerate(metrics):
        if m in means:
            split = list(means[m].keys())[0]
            print(f"{metric_names[i]}: {means[m][split][0] * metric_scales[i] :.2f}")


def remove_outliers(monitor):
    # get inlier indices
    trans = monitor.get("obj_translation_3d")[0]
    valid_idx = trans < 1000.0
    if np.sum(valid_idx) < trans.size:
        print(f"Removing {trans.size - np.sum(valid_idx)} outliers out of {trans.size} samples)")

    # remove outliers
    for m, sdict in monitor.data.items():
        for s, edict in sdict.items():
            assert len(edict.keys()) == 1, "Expecting a monitor with only one epoch per split!"
            for e, value in edict.items():
                if trans.size == value.size:
                    monitor.data[m][s][e] = value[valid_idx]
                else:
                    #print(f"metric {m} split {s} epoch {e} has wrong size. Expected {trans.size} but got {value.size}.")
                    pass

    return monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", "-m", help="Path to the metrics.pkl file that was created during evaluation.")
    args = parser.parse_args()

    result = main(args)