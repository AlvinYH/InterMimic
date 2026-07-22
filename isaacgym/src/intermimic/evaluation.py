import math

import torch


def select_evaluation_motion(valid_motion_ids, execution_steps, visit_count):
    """Choose one object-compatible motion deterministically for evaluation."""

    missing = valid_motion_ids[execution_steps[valid_motion_ids] < 1]
    candidates = missing if missing.numel() > 0 else valid_motion_ids
    visits = visit_count[candidates]
    return candidates[visits == visits.min()].min()


def validate_evaluation_report(report):
    """Reject an internally inconsistent native evaluation report."""

    coverage = report["coverage"]
    success = report["success"]
    metrics = report["metrics"]
    motions = report["motions"]

    total = coverage["total"]
    motion_ids = [motion["motion_id"] for motion in motions]
    if len(motion_ids) != len(set(motion_ids)):
        raise RuntimeError("InterMimic evaluation report has duplicate motion IDs")
    if sorted(motion_ids) != list(range(total)):
        raise RuntimeError("InterMimic evaluation report has missing motion IDs")

    evaluated = [motion for motion in motions if motion["evaluated"]]
    missing_ids = [motion["motion_id"] for motion in motions if not motion["evaluated"]]
    if coverage["evaluated"] != len(evaluated):
        raise RuntimeError("InterMimic evaluation coverage count is inconsistent")
    if coverage["missing_motion_ids"] != missing_ids:
        raise RuntimeError("InterMimic evaluation missing-motion list is inconsistent")
    if coverage["complete"] != (len(missing_ids) == 0):
        raise RuntimeError("InterMimic evaluation completion flag is inconsistent")

    if any(motion["visits"] < 1 for motion in evaluated):
        raise RuntimeError("InterMimic evaluated motion has zero visits")
    success_count = sum(motion["success"] for motion in motions)
    if success["count"] != success_count or success["denominator"] != total:
        raise RuntimeError("InterMimic evaluation success denominator is inconsistent")
    if not math.isfinite(success["rate"]) or not math.isclose(
        success["rate"], success_count / total
    ):
        raise RuntimeError("InterMimic evaluation success rate is inconsistent")

    if metrics["aggregation"] != "evaluated_only":
        raise RuntimeError("InterMimic evaluation metric aggregation is ambiguous")
    if metrics["evaluated_count"] != len(evaluated):
        raise RuntimeError("InterMimic evaluation metric count is inconsistent")
    metric_values = (
        metrics["average_execution_steps"],
        metrics["average_human_pose_error"],
        metrics["average_object_pose_error"],
    )
    if evaluated and not all(
        value is not None and math.isfinite(value) for value in metric_values
    ):
        raise RuntimeError("InterMimic evaluation metrics contain non-finite values")
    if not evaluated and any(value is not None for value in metric_values):
        raise RuntimeError("InterMimic empty evaluation must not report metric means")

    for motion in evaluated:
        values = (
            motion["execution_steps"],
            motion["human_pose_error"],
            motion["object_pose_error"],
        )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("InterMimic motion metrics contain non-finite values")
