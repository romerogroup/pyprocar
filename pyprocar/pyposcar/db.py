from . import dbCovalentBond
import numpy as np

class DB:
  def __init__(self, customRadii = None):
    cr = dbCovalentBond.covalent_radii
    cr = cr.replace('-', 'nan')
    cr_names = [x.split()[1] for x in cr.split('\n')]
    cr_values = [x.split()[2:] for x in cr.split('\n')]
    # converting the values to an array, in Angstroms
    cr_values = [np.array(x, dtype=float)/100 for x in cr_values]
    self.covalentRadii = dict(zip(cr_names,cr_values))
    # print(self.covalentRadii)
    
    if(customRadii != None):
      self.informed_radii = customRadii
    else:
      self.informed_radii = {'':''}

    return
  def estimateBond(self, element1, element2):
    """Estimates the covalent bond by summing the larger covalent radius
    for each atoms
    """

    if(element1 in self.informed_radii.keys()):
      radii1 = self.informed_radii[element1]
    else:  
      radii1 = np.nanmax(self.covalentRadii[element1])

    if(element2 in self.informed_radii.keys()):
      radii2 = self.informed_radii[element2]
    else:  
      radii2 = np.nanmax(self.covalentRadii[element2])

    return radii1 + radii2
  def get_bandwidth(self, element1, element2):
    if((element1 == "H") or (element2 == "H")):
      return 0.02
    else:
      return 0.1
  	
    
    

atomicDB = DB()
