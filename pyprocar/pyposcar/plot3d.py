
import matplotlib.pyplot as plt
import numpy as np
class data3D:
  def __init__(self, data, lattice, verbose='debug'):
    """args:

    `data` is a 3D mesh numpy array. It is understood that their
    domain ranges from 0 to 1

    `lattice` is the matrix to convert the points in `data` from [0,1)
    to their real domain, non-orthogonal lattices are accepted

    `verbose`: verbosity level. Three values are accepted {False, True, 'debug'}

    """
    
    self.data = data
    self.verbose = verbose
    self.lattice = lattice
    if self.verbose == 'debug':
      print('DEBUG: data3D... init()')
      print('DEBUG: data.shape, ', self.data.shape)
    return


  def _get_plane_vectors(self, axis):
    """This method find two vectors perpendicular to a given axis. It
    doesn't matter how axis is chosen (it does matter in the method
    that invokes it)

    """
    # I any need two vectors orthogonal to axis, lets try x,y,z, in that order
    axis = axis/np.linalg.norm(axis)
    x,y,z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    proj_x = np.dot(x,axis)
    proj_y = np.dot(y,axis)
    proj_z = np.dot(z,axis)

    v1 = None
    v2 = None
    # I will choose the least projected cartesian vector, and get the
    # perpendicular (to `axis`) component
    if np.abs(proj_x) <= np.abs(proj_y) and np.abs(proj_x) <= np.abs(proj_z):
      v1 = x - proj_x*axis
    if np.abs(proj_y) <= np.abs(proj_x) and np.abs(proj_y) <= np.abs(proj_z) and v1 is None:
      v1 = y - proj_y*axis
    if np.abs(proj_z) <= np.abs(proj_x) and np.abs(proj_z) <= np.abs(proj_y) and v1 is None:
      v1 = z - proj_z*axis
    v1 = v1/np.linalg.norm(v1)
    
    # the second vector is orthogonal to both, axis
    v2 = np.cross(axis, v1)
    # this should be unnecessary
    v2 = v2/np.linalg.norm(v2)
    
    if self.verbose == 'debug':
      print('DEBUG: projections (x,y,z),', proj_x, proj_y, proj_z)
      print('DEBUG: _get_plane_vectors(), plane vectors')      
      print(v1, v2)
    return v1, v2
    
    
    
    

    
    
  def cut_plane(self, axis, value):
    """
    plot the data along the desired axis at the selected `value`.

    args:

    `axis` can be several things:
      - a vector respect to the `lattice`. `axis` will be normalized

    """
    if self.verbose=='debug':
      print('DEBUG: cut_plane:...')
      print('DEBUG: axis,', axis)
      print('DEBUG: value,', value)
      
      
    from scipy.ndimage.interpolation import map_coordinates
    
    # I need to vectors passing for the plane 
    v1, v2 = self._get_plane_vectors(axis)
    
    # from these points I need to get points to interpolate. Several
    # points will be generated, and these outside of the box will be
    # discarded
    ###
    ### Mind, here I could made two list, one applying PBCs, if I am
    ### interested on that
    ###
    d_shape = self.data.shape
    s_max = np.max(d_shape)
    coord_x = v1*np.linspace(-smax, smax, d_shape*0)

    

    ###
    ### I will disregard anything related to lattices by the moment
    ###
    
    # the axis (let's say `v`) is in a non-orthonormal lattice
    # It has to be changed to an orthonormal lattice (e1, e2, e3) with e1=[1,0,0]
    # In that lattice it will be v', satisfying:
    #
    # lattice * v' = v,  or 
    # lattice^{-1} * v = v'
    #
    # However, for interpolation purposes, a *evenly* sampled
    # orthogonal grid is the rule: it isn't normalized to unity
    # 
    v = axis
    invLatt = np.linalg.inv(self.lattice)
    vp = invLatt*v
    vp = vp/np.linalg.norm(vp)
    # `value` is not scaled by this transformation

    

  def interpolate_data(self, method='nearest'):
    pss
  
  def rot_matrix_2vec(self, axis1, axis2):
    """

    It returns a rotation matrix to rotate `axis1` to `axis2`
    
    """
    if self.verbose == 'debug':
      print('\nDEBUG: data3D.rot_matrix_2vec: ...', )
    # making sure these are unit vectors
    axis1 = np.array(axis1)/np.linalg.norm(axis1)
    axis2 = np.array(axis2)/np.linalg.norm(axis2)
    if self.verbose == 'debug':
      print('DEBUG: axis1 (normalized)', axis1)
      print('DEBUG: axis1 (normalized)', axis1)

    u = np.cross(axis1, axis2)
    u = np.array(u)/np.linalg.norm(u)
    if self.verbose == 'debug':
      print('DEBUG: axis of rotation matrix', u)
    ux, uy, uz = u[0], u[1], u[2]
    uxx, uxy, uxz = ux*ux, ux*uy, uy*uz
    uyx, uyy, uyz = uy*ux, uy*uy, uy*uz
    uzx, uzy, uzz = uz*ux, uz*uy, uz*uz
    
    # angle 'theta' or `T` for shorter
    T = np.arccos(np.dot(axis1, axis2))
    if self.verbose == 'debug':
      print('DEBUG: angle of rotation matrix', u)
    
    cT, sT = np.cos(T), np.sin(T)
    
    R = np.array([[   cT + uxx*(1-cT), uxy*(1-cT) - uz*sT, uxz*(1-cT) + uy*sT],
                  [uyx*(1-cT) + uz*sT,    cT + uyy*(1-cT), uyz*(1-cT) - ux*sT],
                  [uzx*(1-cT) - uy*sT, uzy*(1-cT) + ux*sT,    cT + uzz*(1-cT)]])
    
    if self.verbose == 'debug':
      print('DEBUUG: rotation matrix')
      print(R)
    return R
