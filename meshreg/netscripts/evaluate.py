def parse_evaluators(evaluators, config=None):
    """
    Parse evaluators for which PCK curves and other statistics
    must be computed
    """
    if config is None:
        config = {
            # "joints2d_trans": [0, 50, 20],
            "joints2d_base": [0, 100, 100],
            "corners2d_base": [0, 100, 100],
            "verts2d_base": [0, 100, 100],
            "joints3d_cent": [0, 0.2, 20],
            "joints3d": [0, 0.5, 20],
        }
    eval_results = {}
    for evaluator_name, evaluator in evaluators.items():
        start, end, steps = [config[evaluator_name][idx] for idx in range(3)]
        (epe_mean, epe_mean_joints, epe_median, auc, pck_curve, thresholds) = evaluator.get_measures(
            start, end, steps
        )
        eval_results[evaluator_name] = {
            "epe_mean": epe_mean,
            "epe_mean_joints": epe_mean_joints,
            "epe_median": epe_median,
            "auc": auc,
            "thresholds": thresholds,
            "pck_curve": pck_curve,
        }
    return eval_results
