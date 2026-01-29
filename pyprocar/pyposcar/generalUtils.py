from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def remove_flat_points(
    samples: npt.NDArray[np.floating[Any]],
    scores: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """To use scipy.argrelmin, or other similar method, the array needs no
    to be flat in left or right directions, otherwise it will fail to
    detect the minima

    This method removes the points with consecutive same `score`, from
    `samples` and `scores`.

    `samples` must be ordered

    """
    # print('\n remove_flat_points')
    # print(samples)
    # print(scores)
    last_score: np.floating[Any] = scores[0]
    new_samples: list[np.floating[Any]] = []
    new_scores: list[np.floating[Any]] = []
    new_samples.append(samples[0])
    new_scores.append(scores[0])

    for i in range(1, len(samples)):
        if scores[i] != last_score:
            new_samples.append(samples[i])
            new_scores.append(scores[i])
        last_score = scores[i]
    return (np.array(new_samples), np.array(new_scores))
