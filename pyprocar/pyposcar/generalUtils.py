
import numpy as np

def remove_flat_points(samples, scores):
  """To use scipy.argrelmin, or other similar method, the array needs no
  to be flat in left or right directions, otherwise it will fail to
  detect the minima

  This method removes the points with consecutive same `score`, from
  `samples` and `scores`.

  `samples` must be ordered 
  
  """
  #print('\n remove_flat_points')
  #print(samples)
  #print(scores)
  last_score = scores[0]
  new_samples, new_scores = [], []
  new_samples.append(samples[0])
  new_scores.append(scores[0])
  
  for i in range(1, len(samples)):
    if scores[i] != last_score:
      new_samples.append(samples[i])
      new_scores.append(scores[i])
    last_score = scores[i]
  return (np.array(new_samples), np.array(new_scores))
