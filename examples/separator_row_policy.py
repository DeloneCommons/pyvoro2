"""Independent separator units and bounded row policy without tessellation."""

import numpy as np

from pyvoro2.inverse.separator import (
    ExponentialBoundaryPenalty, FitModel, Interval, L2Regularization,
    ReciprocalBoundaryPenalty, SoftIntervalPenalty, SquaredLoss,
    dumps_report_json, fit_weights_from_separators, resolve_separator_observations,
)


def mixed_policy_report():
    points = np.array([[0., 0.], [2., 0.], [4., 0.]])
    observations = resolve_separator_observations(
        points, [(0, 1, .25), (1, 2, .5), (0, 2, .5)],
    )
    model = FitModel(
        mismatch=SquaredLoss(space='position'),
        feasible=Interval([0., .5, 0.], [1., .5, 1.],
                          applicable=[True, True, False], space='fraction'),
        penalties=(SoftIntervalPenalty(.375, .625, [8., 0., 0.]),
                   ExponentialBoundaryPenalty(strength=0., space='position'),
                   ReciprocalBoundaryPenalty(strength=[0., 0., 0.])),
        regularization=L2Regularization(0., [1., 2., 3.]),
    )
    result = fit_weights_from_separators(
        points, observations, model=model, solver='admm', admm_max_iter=5000,
    )
    if result.status != 'optimal':
        raise RuntimeError(result.status_detail)
    return result.to_report(observations)


if __name__ == '__main__':
    print(dumps_report_json(mixed_policy_report(), sort_keys=True))
