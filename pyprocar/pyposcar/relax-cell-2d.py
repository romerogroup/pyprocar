#!/usr/bin/env python3

import poscarUtils
import poscar
import os
import argparse
import re
import numpy as np
import sys

def get_data(old=12, max_iterations=100):
  """Getting the information from two sources:
  1) The 'OPT_SUM' file, if it exists it has the energies and lattice
  data of previous iterations

  2) The 'OUTCAR' file, it is useful to guess a first step if there
  is no enough data to fit a parabola
  
  The OPT_SUM file will be updated (and created if needed).
  This file contains:

  lat_a lat_b angle_ab energy stress_a stress_b stress_theta
  
  args:
  `old`: discard history data older than this iteration
  `max_iterations`: if reached the calculation stops
  """

  if not os.path.isfile('OUTCAR'):
    raise RuntimeError('File OUTCAR does not exist')
  outcar = open('OUTCAR', 'r').read()
  energy = re.findall(r'energy without entropy =\s*([-.\d]+)', outcar)
  energy = float(energy[-1])
  print('energy, ', energy)

  stress = re.findall(r'Total[-.\s\d]+\s*in kB', outcar)
  stress = stress[-1].split()
  stress_A = float(stress[1])
  stress_B = float(stress[2])
  stress_theta = float(stress[4])
  print('stress A,', stress_A)
  print('stress B,', stress_B)
  print('stress theta,', stress_theta)

  # locating the parameters
  vectors = re.findall(r'reciprocal lattice vectors[-.\d\s]*', outcar)
  vectors = vectors[-1].split()
  vec_a = np.array(vectors[3:6], dtype=float)
  vec_b = np.array(vectors[9:12], dtype=float)
  print('2D lattice, ', vec_a, vec_b)
  length_a = np.linalg.norm(vec_a)
  length_b = np.linalg.norm(vec_b)
  print('length A, ', length_a)
  print('length B, ', length_b)
  angle_ab = np.arccos(np.dot(vec_a, vec_b)/length_a/length_b)
  print('angle AB', angle_ab*180/np.pi, '(deg)', ', ', angle_ab, ' (rad)')
  
  
  # updating and perhaps creating a file with energies
  line = ''
  line += str(length_a) + ' '
  line += str(length_b) + ' '
  line += str(angle_ab) + ' '
  line += str(energy) + ' '
  line += str(stress_A) + ' '
  line += str(stress_B) + ' '
  line += str(stress_theta) + '\n'
  
  
  print('New line to OPT_SUM\n',line)
  f_histo = open('OPT_SUM', 'a')
  f_histo.write(line)
  f_histo.close()

  # reading the full history
  history = open('OPT_SUM', 'r').readlines()
  print('number of history points', len(history))
  if len(history) > max_iterations:
    continue_loop(False)
    raise RuntimeError('Maximum number of iteration reached')
  if len(history) > old:
    history = history[-old:]
    print('Deleting old values from history')
  history = list(set(history))
  history = [x.split() for x in history]
  history = np.array(history, dtype=float)
  print('History (excluding duplicate points, unsorted)')
  print(history)

  # passing the relevant info:
  return history

def make_prediction_a(history, maxStep=0.01):
  raise NotImplementedError

def make_prediction_b(history, maxStep=0.01):
  raise NotImplementedError

def make_prediction_a_and_b(history, maxStep=0.01):
  raise NotImplementedError

def make_prediction_theta(history, maxStep=0.01):
  raise NotImplementedError



def make_prediction_ab(history, tolerance=0.1, maxStep=0.01):
  """`history` is an array with the form:
  [[lat_a lat_b angle_ab energy stress_a stress_b stress_theta],
   [...],
   [...]]
  
  maxStep: relative maximum change. Its default, 0.01 is 1%

  tolerance: maximum value (absolute) of stress to consider the
  minimization achieved. In eV/A

  return value:
  True: if new step was predicted
  False: if convergence has been achieved

  """
  print('\n\nStarting prediction:\n')

  # filtering only relevant data from history. The stress is the
  # average value.
  stress = (history[:,4] + history[:,5])/2
  param_a = history[:,0]
  param_b = history[:,1]
  energy = history[:,3]

  # looking if the convergence has been achieved
  if np.min(np.abs(stress)) <= tolerance:
    print('Convergence achieved!')
    return False

  # with one data points, there is nothing to interpolate. Just
  # moving a little bit from the lowest energy point
  if len(history) == 1:
    e = energy[0] # just one data point
    s = stress[0] # just one data point
    print('Initial point, initial stress', s, 'initial parameters,', param_a, param_b)
    if s < 0: # negative stress -> decrease the unit cell
      new_param_a = param_a[0]*(1 - maxStep)
      new_param_b = param_b[0]*(1 - maxStep)      
    else:
      new_param_a = param_a[0]*(1 + maxStep)
      new_param_b = param_b[0]*(1 + maxStep)
  # I will handle the cases of 2, 3 or more data points together
  elif len(history) >= 2:
    i_min, i_max = np.argmin(energy), np.argmax(energy)
    # The fitting of the parabola depends on how many data points
    if len(history) == 2:
      print('There are two data points...')
      # I will fit them an parabola with the next conditions:
      # 1 passing by (param_a[0], energy[0]) 
      # 2 passing by (param_a[1], energy[1])
      # the ratio of the derivatives at `param_a[0], param_a[1]` is the ratio
      # between the stress values.
      parabola = parabola_two_points_derivative(param_a[0], energy[0], stress[0],
                                                param_a[1], energy[1], stress[1])
    elif len(history) == 3:
      print("There are three data points...")
      parabola = parabola_three_points(param_a[0], energy[0],
                                       param_a[1], energy[1],
                                       param_a[2], energy[2],)
    else:
      parabola = fit_parabola_1D_close_min(param_a, energy)
    # y = a*x^2 + b*x + c
    a, b, c = parabola[0], parabola[1], parabola[2]
    # the minimum of the parabola is 
    new_param_a = -b/2/a

    min_energy = a*new_param_a**2 + b*new_param_a + c
    print('Fitting a parabola yield (param_a_min, E_min)', new_param_a, min_energy)
    if a<0:
      print('\nThe parabola is inverted!\n, falling back to a simple prediction\n')
      if stress[i_min] < 0: # negative stress -> decrease the unit cell
        new_param_a = param_a[i_min]*(1 - maxStep)
        new_param_b = param_b[i_min]*(1 - maxStep)      
      else:
        new_param_a = param_a[i_min]*(1 + maxStep)
        new_param_b = param_b[i_min]*(1 + maxStep)


    # if stress is negative param_a has to decrease
    if stress[i_min] < 0:
      # I will keep the smaller decrement (larger value) among the
      # predicted minimum and the maxStep
      new_param_a = max(param_a[i_min]*(1 - maxStep), new_param_a)
    else:
      new_param_a = min(param_a[i_min]*(1 + maxStep), new_param_a)
    # fitting param_b from new_param_a
    factor = new_param_a/param_a[i_min]
    new_param_b = param_b[i_min]*factor

  print('New parameter a,', new_param_a)
  print('New parameter b,', new_param_b)

  #####
  ##### Caution, the factors should be applied on the MINIMUM values,
  ##### not on the current POSCAR file
  #####
  p = poscar.Poscar('CONTCAR')
  p.parse()
  lat_a = np.linalg.norm(p.lat[0])
  factor = new_param_a/lat_a
  factor = np.array([factor, factor, 1.0])
  print('factors to modify current POSCAR:', factor )
  new_p = poscarUtils.poscar_modify(p)
  new_p.scale_lattice(factor, cartesian=False)
  new_p.write('POSCAR-NEW')

  return True

def parabola_two_points_derivative(x1, y1, dy1, x2, y2, dy2):
  """It calculates the of a parabola passing by (x1, y1) and (x2, y2),
  and with derivatives `dy1`, `dy2` 
  """
  A = np.array([[2*(dy1*x2-dy2*x1), dy1-dy2, 0],
                [            x1*x1,      x1, 1],
                [            x2*x2,      x2, 1]])
  C = np.array([[0],
                [y1],
                [y2]])
  #print('The matrix A is')
  #print(A)
  #print('The matrix C is')
  #print(C)
  X = np.dot(np.linalg.inv(A), C)
  X.shape = (3)
  #print('the solution is,' )
  #print(X)
  return X

def parabola_three_points(x1, y1, x2, y2, x3, y3):
  """It calculates the of a parabola passing by (x1, y1) , (x2, y2) and
  (x3, y3)

  """
  A = np.array([[x1*x1, x1, 1],
                [x2*x2, x2, 1],
                [x3*x3, x3, 1]])
  C = np.array([[y1],
                [y2],
                [y3]])
  X = np.dot(np.linalg.inv(A), C)
  X.shape = (3)
  return X


def fit_parabola_1D(x,y):
  # 
  import scipy.optimize
  def parabola(x, a, b, c):
    return a*x*x + b*x + c
  fit_params, pcov = scipy.optimize.curve_fit(parabola, x, y)
  #print('fitting,', fit_params)
  return fit_params

def continue_loop(value):
  if value == True:
    # convergence has NOT been achieved
    open('RELAX-CONTINUE', 'a').close()
  else:
    try:
      os.remove('RELAX-CONTINUE')
    except FileNotFoundError:
      pass

    

def fit_parabola_1D_close_min(x,y, max_data=6):
  """It fits a parabola with all the data, locates the minimum, and then
  it keeps up to `max_data` data points, discarding those farther from
  the minimum. 

  In the rare case there are points rqually far from the
  minimum, both points are kept to re-calculate the parabola

  return:
  [A,B,C] # A*x*x + B*x + C = y

  """
  x = np.array(x) # just in case
  params = fit_parabola_1D(x,y)
  a, b, c = params[0], params[1], params[2]
  x_min = -b/2/a
  distances = np.abs((x-x_min))

  if len(x) <= max_data:
    return params
  else:
    d_cutoff = np.sort(distances)[max_data]
    new_x = []
    new_y = []
    for i in range(len(x)):
      if distances[i] <= d_cutoff:
        new_x.append(x[i])
        new_y.append(y[i])
    params = fit_parabola_1D(x,y)
    return params
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='A lattice optimizer for a 2D'
                                    ' lattice. It will look to the old data and'
                                    ' try to fit a parabola (1D or 2D) on it')

  parser.add_argument('--constrain', '-c',  help='What has to to be optimized? '
                      'just `a`?, just `b`?, `ab` (both a,b keeping the aspect '
                      'ratio)?, the angle `theta`?, or both `a_and_b` (varied '
                      'independently)', choices=['a','b', 'ab', 'theta'],
                      default='ab')
  parser.add_argument('-tolerance', '-t', type=float, help='maximum stress allowed'
                      ' (absolute value, in eV/A)', default=0.1)
  parser.add_argument('-s', '--script', action='store_true', help='It creates a '
                      'sample snippet of a script to run the relaxation. Then quit')
  parser.add_argument('-o', '--discard_old', default=12, help='Discard data older'
                      ' than this value, in iterations')
  parser.add_argument('-m', '--max_iterations', default=100, help='Max number of'
                      ' iterations, before failing')
  args = parser.parse_args()

  if args.script:
    string = '#!/usr/bin/env bash\n\n'
    string += 'VASPCOMMAND="mpirun -np 24 vasp_std" # e.g. mpirun -np 12 vasp_std\n'
    string += 'RELAXCOMMAND="relax-cell-2d.py" # the actual command to run this code\n\n'
    # I need to have a vasp_simulation to start
    string += 'if [ ! -f OUTCAR ]; then\n'
    string += '    echo "OUTCAR file does not exists. Running VASP"\n'
    string += '    $VASPCOMMAND\n'
    string += 'fi\n\n'
    # I need at least one point to decide if iterate
    string += '$RELAXCOMMAND\n\n'
    # while RELAX-CONTINUE exists iterate
    string += 'while [ -f RELAX-CONTINUE ]\n'
    string += 'do\n'
    string += '    rm RELAX-CONTINUE\n' # just in case, I don't want to iterate forever
    string += '    cp POSCAR-NEW POSCAR\n'
    string += '    $VASPCOMMAND\n'
    string += '    $RELAXCOMMAND\n'
    string += 'done\n'
    string += 'cp CONTCAR POSCAR'
    f = open('sample-script.sh', 'w')
    f.write(string)
    f.close()
    sys.exit()
  
  history = get_data(old=args.discard_old, max_iterations=args.max_iterations)
  if args.constrain == 'a':
    prediction = make_prediction_a(history, args.tolerance)
  if args.constrain == 'b':
    prediction = make_prediction_b(history, args.tolerance)
  if args.constrain == 'ab':
    prediction = make_prediction_ab(history, args.tolerance)
  if args.constrain == 'a_and_b':
    prediction = make_prediction_a_and_b(history, args.tolerance)
  if args.constrain == 'theta':
    prediction = make_prediction_theta(history, args.tolerance)

  continue_loop(prediction)
  

