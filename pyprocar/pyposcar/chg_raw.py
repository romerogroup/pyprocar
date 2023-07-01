#!/usr/bin/env python3

######################
## TODO:
## -Chg.shift 
## -Chg.plot_atoms (maybe in poscar.py)
## -Chg.Charge_redistributions
######################

import argparse
import os
import numpy as np
import poscar
import re
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings
import plot3d

class Chg_base:
  def __init__(self):
    self.comment = None
    self.Ispin = 1
    self.NGF = []
    self.poscar = None
    self.File = []
    self.Data_blocks = []
    self.Data0 = np.array([])
    self.Data1 = np.array([])
    self.Data2 = np.array([])
    self.Data3 = np.array([])
    self.is_chg = True
    self.is_locpot = None
    
  def Load(self, filename='CHGCAR', frame=0, is_chg=None, verbose=None):
    """Load a CHG-like file 

    `verbose` = False: No verbosity
              = True: verbose output
              = 'debug': usually unwanted verbosity level

    """
    if verbose != None:
      self.verbose = verbose
    if is_chg != None:
      self.is_chg = is_chg
    if self.verbose:
      print("\nINFO: Loading a CHG-like with the following parameters:")
      print("INFO: Filename: ", filename)
      print("INFO: is_chg:   ", self.is_chg)
    # checking if the file exist
    if not os.path.isfile(filename):
      print("ERROR: can't open the file, please check:", filename)
      raise RuntimeError('File does not exist')
    self.File = open(filename, 'r').read()
        
    # getting how many ionic steps were loaded
    self.comment = re.findall(r'^[^\n]*\n', self.File)[0]
    if self.verbose == 'debug':
      print('DEBUG: Comment line:')
      print(self.comment)

    Nframes = len(re.findall(self.comment, self.File))
    if self.verbose == 'debug':
      print('DEBUG: Number of frames found:', Nframes)

    if self.verbose:
      print('INFO: Selecting the frame:', frame)
    frames = re.split(self.comment, self.File)
    # if the first frame is empty (a `re` thing), should be discarded
    if len(frames[0]) == 0:
      frames.pop(0)
    self.File = frames[frame]

    # next string to find is the line with
    # NGFx NGFy NGFz
    NGFline = re.findall(r'\n\s*\n(\s+\d+\s+\d+\s+\d+)\n', self.File)
    if verbose:
      print('INFO: NGFx NGFy NGFz', NGFline)
    if len(NGFline) != 1:
      raise RuntimeError('NGFline should have one and only one occurrence')
    NGFline = NGFline[0]
    self.NGF = np.array(NGFline.split(), dtype=int)
    if verbose == 'debug':
      print('DEBUG: grid size,', self.NGF)

    data = re.split(NGFline, self.File)
    ndata = len(data)
    if self.verbose:
      print('INFO: number of data blocks', ndata)
    
    # The first block is a POSCAR-like string
    # The second block is the spin-up
    # The third block (if present) is the spin-down
    # The 4th and 5th blocks are Sy, Sz (1s:rho, 2nd:Sx)
    if ndata == 2:
      self.Ispin = 1
      print('\nA non-magnetic calculation detected\n')
    elif ndata == 3:
      self.Ispin = 2
      print('\nA spin-polarized calculation detected\n')
    elif ndata == 5:
      self.Ispin = 4
      print('\nA non-collinear calculation detected\n')
    else:
      raise RuntimeError('Number of block data is unexpected,' + str(ndata))

    # The poscar-like string would be passed to a POSCAR class
    poscarString = self.comment + data.pop(0)
    # print(poscarString)
    self.poscar = poscar.Poscar(filename=None)
    self.poscar.parse(fromString=poscarString)
    if self.verbose == 'debug':
      print('DEBUG: POSCAR-like info:')
      print('\n'.join(self.poscar.poscar))
    
    # Now we will search for augmentation charges
    # 'augmentation occupancies'
    # That info will be discarded
    temp = []
    for block in data:
      temp.append(re.split(r'augmentation occupancies', block)[0])
    data = temp
    
    # The grid data will be processed
    Ndata = self.NGF[0]*self.NGF[1]*self.NGF[2]
    if self.verbose == 'debug':
      print('DEBUG: Data points expected:', Ndata)
      
    self.Data0 = data[0].split()
    ngfx, ngfy, ngfz = self.NGF[0], self.NGF[1], self.NGF[2]

    # If the grid points don't agree it could be a LOCPOT with residual data 
    if len(self.Data0) != Ndata:
      N = self.poscar.Ntotal
      if len(self.Data0) == Ndata + N:
        self.is_locpot = True
        self.Data0 = np.array(self.Data0[:-N], dtype=float).reshape(ngfz,ngfy,ngfx)
        if self.verbose == 'debug':
          print('INFO: a LOCTOP file was detected')
        if self.is_chg and self.is_locpot:
          if self.verbose == 'debug':
            print('DEBUG: The number of grid points is not what I was expecting. '
                  'The data I got is:')
            print(self.Data0[:30])
            print(self.Data0[-30:])
          raise RuntimeError('The file is flagged as a CHGCAR-like file'
                             ' and as a LOCPOT at the same time. This is inconsistent')
      else:
        raise RuntimeError('Grid points do not agree')
    else:
      self.Data0 = np.array(self.Data0, dtype=float).reshape(ngfz,ngfy,ngfx)

    if self.is_chg:
      self.Data0 = self.Data0 / Ndata
      if self.verbose:
        print('Total charge', np.sum(self.Data0))
      
    if self.Ispin > 1:
      self.Data1 = data[1].split()
      if len(self.Data1) != Ndata:
        raise RuntimeError('Grid points do not agree')
      self.Data1 = np.array(self.Data1, dtype=float).reshape(ngfz,ngfy,ngfx)
      if self.is_chg:
        self.Data1 = self.Data1 / Ndata
        if self.verbose and self.Ispin == 2:
          print('INFO: total magnetization,', np.sum(self.Data1))

    if self.Ispin == 4:
      self.Data2 = data[2].split()
      self.Data3 = data[3].split()
      if len(self.Data2) != Ndata or len(self.Data3) != Ndata:
        raise RuntimeError('Grid points do not agree')
      self.Data2 = np.array(self.Data2, dtype=float).reshape(ngfz,ngfy,ngfx)
      self.Data3 = np.array(self.Data3, dtype=float).reshape(ngfz,ngfy,ngfx)
      if self.is_chg:
        self.Data2 = self.Data2 / Ndata
        self.Data3 = self.Data3 / Ndata
  

class Chg:
  def __init__(self, filename='CHG', is_chg=True, verbose=False):
    self.chg = Chg_base()
    self.filename = filename
    self.is_chg = is_chg
    self.verbose = verbose
    self.chg.Load(filename=self.filename,
                  frame=0,
                  is_chg=self.is_chg,
                  verbose=self.verbose)

  def Zplot(self, level=None, spin=0, cart_level=None, direct_level=None):
    """it plots the CHG-like file at an specific z-value, given by
    level. Only works properly when the Z-axis (c-vector) is perpendicular to the
    other axes.

    args:

    spin: the spin channel (i.e. the first `0`, or the second `1`
    entry of the file)

    level: the value of z to plot. In terms of the grid values (see
    NGF in OUTCAR). Only one among `level`, `cart_level`, and
    `direct_level` should be provided.

    cart_level: as `level`, but the value of z is in cartesian.

    direct_level: as `level`, but the value is in direct coordinates.

    """
    if (level and cart_level) or (level and direct_level) or (direct_level and cart_level):
      raise RuntimeError('only one among `level`, `direct_level`, and'
                         ' `cart_level` has to be provided')
    # setting the different kind of levels in order, `cart_level` sets
    # `direct_level` and so on
    if cart_level:
      c = np.linalg.norm(self.chg.poscar.lat[2])
      direct_level = cart_level/c
      if self.verbose:
        print('INFO: cart_level,', cart_level)
    if direct_level:
      # going to a grid level, crude interpolation of the level
      level_float = direct_level*self.chg.NGF[2]
      level_floor = math.floor(level_float)
      level_ceil = math.ceil(level_float)
      delta_floor = 1-np.abs(level_float - level_floor)
      delta_ceil = 1-np.abs(level_ceil - level_float)
      if level_floor == level_ceil:
        delta_floor, delta_ceil = 1,0
      if self.verbose == 'debug':
        print('INFO: direct_level', direct_level)
        #print('level_float', level_float)
        #print('level_floor, level_floor', level_floor, level_floor)
        #print('delta_floor,delta_ceil', delta_floor,delta_ceil)
    if level:
      # in this case the crude interpolation shouldn't do anything
      level_floor = level
      level_ceil = level
      delta_floor, delta_ceil = 0,1
      if self.verbose == 'debug':
        print('INFO: level', level)
    if level is None and cart_level is None and direct_level is None:
      level_floor = self.chg.NGF[2]/2
      level_ceil = level_floor
      delta_floor, delta_ceil = 0,1
      if self.verbose == 'debug':
        print('INFO: default is the midpoint of the axis', level_floor)
      
    if spin == 0:
      data = self.chg.Data0
    elif spin == 1:
      data = self.chg.Data1
    elif spin == 2:
      data = self.chg.Data2
    elif spin == 3:
      data = self.chg.Data3
    else:
      raise RuntimeError('No such spin channel, ' + str(spin))
    
    zcolor = data[level_floor]*delta_floor + data[level_ceil]*delta_ceil
      
    Agrid, Bgrid = np.mgrid[0:1:self.chg.NGF[0]*1j, 0:1:self.chg.NGF[1]*1j]
    #cartesian value of each point of the grid
    xgrid = Agrid*self.chg.poscar.lat[0,0] + Bgrid*self.chg.poscar.lat[1,0]
    ygrid = Agrid*self.chg.poscar.lat[0,1] + Bgrid*self.chg.poscar.lat[1,1]
    if self.verbose == 'debug':
      print('DEBUG: Agrid.shape', Agrid.shape)
      print('DEBUG: lattice\n', self.chg.poscar.lat)
      print('DEBUG: xgrid.shape', xgrid.shape)

    xmin, xmax = xgrid.min(), xgrid.max()
    ymin, ymax = ygrid.min(), ygrid.max()
    xi, yi = np.mgrid[xmin:xmax:self.chg.NGF[0]*4j, ymin:ymax:self.chg.NGF[1]*4j]
    points = np.vstack((xgrid.flatten(), ygrid.flatten())).T
    values = zcolor.flatten()
    if self.verbose == 'debug':    
      print('DEBUG: points.shape', points.shape)
      print('DEBUG: values.shape', values.shape)
      print('DEBUG: xi.shape', xi.shape)
      print('DEBUG: yi.shape', yi.shape)
    zi = griddata(points, values, (xi, yi), method='linear')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1= ax.pcolormesh(xi, yi, zi, cmap='seismic')
    fig.colorbar(p1)
    ax.set_aspect('equal')
    return fig

  def CutPlot(self, level=None, spin=0, cart_level=None, direct_level=None, axis='c'):
    """it plots the CHG-like file at an specific value of `axis`, given by
    `level`. Only works properly when the selected `axis` (basis vector) is perpendicular
    to the other axes.

    args:

    spin: the spin channel (i.e. the first `0`, or the second `1`
    entry of the file)

    level: the value of x to plot. In terms of the grid values (see
    NGF in OUTCAR). Only one among `level`, `cart_level`, and
    `direct_level` should be provided.

    cart_level: as `level`, but the value of z is in cartesian.

    direct_level: as `level`, but the value is in direct coordinates.

    axis: 'a', 'b', 'c', the basis vector to fix its value

    """
    # an utilitary dict to choose the desired axes
    ax_dict = {'a':0, 'b':1, 'c':2}
    # axis 3 is the axis to make the cut
    ax3 = ax_dict[axis]
    ax1 = np.remainder(ax_dict[axis]+1, 3)
    ax2 = np.remainder(ax_dict[axis]+2, 3)
    if self.verbose:
      print('INFO: Selected axis to cut:', ax3)
    if self.verbose=='debug':
      print('DEBUG: The other axes are:', ax1, ax2)

    
    if (level and cart_level) or (level and direct_level) or (direct_level and cart_level):
      raise RuntimeError('only one among `level`, `direct_level`, and'
                         ' `cart_level` has to be provided')
    # setting the different kind of levels in order, `cart_level` sets
    # `direct_level` and so on
    if cart_level:
      # length of the perpendicular vector
      L = np.linalg.norm(self.chg.poscar.lat[ax3])
      direct_level = cart_level/L
      if self.verbose:
        print('INFO: cart_level,', cart_level)
    if direct_level:
      # going to a grid level, crude interpolation of the level
      level_float = direct_level*self.chg.NGF[ax3]
      level_floor = math.floor(level_float)
      level_ceil = math.ceil(level_float)
      delta_floor = 1-np.abs(level_float - level_floor)
      delta_ceil = 1-np.abs(level_ceil - level_float)
      if level_floor == level_ceil:
        delta_floor, delta_ceil = 1,0
      if self.verbose:
        print('INFO: direct_level', direct_level)
        # print('level_float', level_float)
        # print('level_floor, level_floor', level_floor, level_floor)
        # print('delta_floor,delta_ceil', delta_floor,delta_ceil)
    if level:
      # in this case the crude interpolation shouldn't do anything
      level_floor = level
      level_ceil = level
      delta_floor, delta_ceil = 0,1
      if self.verbose == 'debug':
        print('INFO: level', level)

    if level is None and cart_level is None and direct_level is None:
      level_floor = self.chg.NGF[ax3]/2
      level_ceil = level_floor
      delta_floor, delta_ceil = 0,1
      if self.verbose == 'debug':
        print('INFO: default is the midpoint of the axis', level_floor)

    if spin == 0:
      data = self.chg.Data0
    elif spin == 1:
      data = self.chg.Data1
    elif spin == 2:
      data = self.chg.Data2
    elif spin == 3:
      data = self.chg.Data3
    else:
      raise RuntimeError('No such spin channel, ' + str(spin))

    # building a grid with existent points
    if axis == 'c':
      zcolor = data[level_floor]*delta_floor + data[level_ceil]*delta_ceil
      # a regular orthogonal grid
      Agrid, Bgrid = np.mgrid[0:1:self.chg.NGF[0]*1j, 0:1:self.chg.NGF[1]*1j]
      # cartesian grid
      xgrid = Agrid*self.chg.poscar.lat[0,0] + Bgrid*self.chg.poscar.lat[1,0]
      ygrid = Agrid*self.chg.poscar.lat[0,1] + Bgrid*self.chg.poscar.lat[1,1]
  
    elif axis == 'b':
      zcolor = data[:,level_floor,:]*delta_floor + data[:,level_ceil,:]*delta_ceil
      Agrid, Bgrid = np.mgrid[0:1:self.chg.NGF[2]*1j, 0:1:self.chg.NGF[0]*1j]
      xgrid = Agrid*self.chg.poscar.lat[2,2] + Bgrid*self.chg.poscar.lat[0,2]
      ygrid = Agrid*self.chg.poscar.lat[2,0] + Bgrid*self.chg.poscar.lat[0,0]

    elif axis == 'a':
      zcolor = data[:,:,level_floor]*delta_floor + data[:,:,level_ceil]*delta_ceil
      Agrid, Bgrid = np.mgrid[0:1:self.chg.NGF[1]*1j, 0:1:self.chg.NGF[2]*1j]
      # xgrid = Agrid*self.chg.poscar.lat[1,1] + Bgrid*self.chg.poscar.lat[1,2]
      # ygrid = Agrid*self.chg.poscar.lat[2,1] + Bgrid*self.chg.poscar.lat[2,2]
      xgrid = Agrid*self.chg.poscar.lat[1,1] 
      ygrid = Bgrid*self.chg.poscar.lat[2,2] + Agrid*self.chg.poscar.lat[1,0]

    if self.verbose == 'debug':
      print('DEBUG: zcolor.shape', zcolor.shape)
      print('DEBUG: Agrid.shape', Agrid.shape)
      print('DEBUG: Bgrid.shape', Bgrid.shape)
      print('DEBUG: lattice\n', self.chg.poscar.lat)
      print('DEBUG: xgrid.shape', xgrid.shape)
      print('DEBUG: ygrid.shape', xgrid.shape)
    
    xmin, xmax = xgrid.min(), xgrid.max()
    ymin, ymax = ygrid.min(), ygrid.max()
    # creating a regular grid to interpolate (the `4` is the resolution)
    xi, yi = np.mgrid[xmin:xmax:self.chg.NGF[ax1]*4j, ymin:ymax:self.chg.NGF[ax2]*4j]
    points = np.vstack((xgrid.flatten(), ygrid.flatten())).T
    values = zcolor.flatten()
    if self.verbose == 'debug':
      print('points.shape', points.shape)
      print('values.shape', values.shape)
      print('xi.shape', xi.shape)
      print('yi.shape', yi.shape)
    zi = griddata(points, values, (xi, yi), method='linear')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1= ax.pcolormesh(xi, yi, zi, cmap='seismic')
    fig.colorbar(p1)
    ax.set_aspect('equal')
    return fig
    
  def shift(self, x=None, y=None, z=None):
    pass
  def plot_atoms(self):
    postitions = self.chg.poscar.cpos
    pass

  def plot_cut_new(self, axis, value):
    """Temporary function to use the plot3D class... very limited
    functionality for now

    """
    p3d = plot3d.data3D(data=self.chg.Data0, lattice=self.chg.poscar.lat, verbose='debug')
    p3d.cut_plane(axis=np.array([0.0, 0.0, 1.0]), value=10)
    pass
  
  def average(self, axis):
    data = self.chg.Data0
    if axis == 'c':
      data = np.average(data, axis=2)
      data = np.average(data, axis=1)
      length = np.linalg.norm(self.chg.poscar.lat[2])
    if axis == 'b':
      data = np.average(data, axis=2)
      data = np.average(data, axis=0)
      length = np.linalg.norm(self.chg.poscar.lat[1])
    if axis == 'a':
      data = np.average(data, axis=1)
      data = np.average(data, axis=0)
      length = np.linalg.norm(self.chg.poscar.lat[0])
    print('Averaged Data shape, ', data.shape)
    x = np.linspace(0, length, len(data))
    plt.plot(x, data)
    plt.show()
    
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("inputfile", type=str,
                      help="input file (CHGCAR, CHG, ELFCAR or LOCPOT)")
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Even more verbosity')
  parser.add_argument('-n', '--no_scale', action='store_true',
                      help='Set when openning a ELFCAR or LOCPOT. The CHGCAR'
                      ' uses a different normalization, this flag avoids it.')
  parser.add_argument('-a', '--axis', choices=['a','b','c'], default='c',
                      help='Axis to cut')
  parser.add_argument('-z', action='store_true', help='Fallback utility function to plot')

  parser.add_argument('--new', action='store_true', help='usage of new, not fully'
                      ' tested methods')
  parser.add_argument('-p', '--average', action='store_true', help='averages the '
                      'potential (or charge) along a given axis')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('-l', '--direct_level', type=float,
                      help='level to plot, in direct coords')
  group.add_argument('-c', '--cartesian_level', type=float,
                      help='level to plot, in cartesian coords')
  

  args = parser.parse_args()

  
  if args.debug:
    args.verbose = 'debug'

  if 'POT' in args.inputfile or 'ELF' in args.inputfile:
    if args.no_scale == False:
      warnings.warn("It seems you are openning a LOCPOT or ELFCAR file."
                    " If so, you should add the option '-n' to get the "
                    "correct scaling")
  elif 'CHG' in args.inputfile and args.no_scale == True:
    warnings.warn("It seems you are openning a CHG or CHGCAR file."
                  " If so, you should not add the option '-n' to get the "
                  "correct scaling")
  is_chg = not args.no_scale
      
  
  chg = Chg(filename=args.inputfile, is_chg=is_chg, verbose=args.verbose)

  
  
  if args.z:
    fig = chg.Zplot(cart_level=args.cartesian_level, direct_level=args.direct_level)
  elif args.new:
    chg.plot_cut_new(value=args.direct_level, axis=args.axis)
  elif args.average:
    chg.average(axis=args.axis)
  else:
    fig = chg.CutPlot(cart_level=args.cartesian_level,
                      direct_level=args.direct_level,
                      axis=args.axis)
  plt.show()
  
