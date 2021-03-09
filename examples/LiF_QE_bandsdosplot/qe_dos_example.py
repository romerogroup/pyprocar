# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 23:19:23 2021

@author: lllang
"""

import pyprocar

#pyprocar.bandsplot(code='qe', elimit=[-15,15], mode='plain')

#pyprocar.bandsplot(code='qe',  elimit=[-15,15], cmap='jet',vmax = 1, vmin = 0 ,mode='parametric',orbitals=[4,5,6,7,8])

pyprocar.dosplot(code='qe',
                  mode='stack_species',
                  # mode='plain',
                  orientation='horizontal',
                  elimit=[-15, 15])
                  # plot_total=True)
# pyprocar.dosplot(code='qe', mode='plain', elimit=[-15,15])


# pyprocar.dosplot(code='qe', mode='parametric', elimit=[-15,15], orbitals=[1,2,3])# plot_total=False)

# pyprocar.dosplot(code='qe', mode='parametric', elimit=[-15,15], orbitals=[1,2,3])

pyprocar.bandsdosplot(code = "qe",
                      bands_mode='plain',
                      dos_mode='plain',
                      # dos_labels=[r'$\uparrow$',r'$\downarrow$'],
                      elimit=[-20,20],)


pyprocar.dosplot(code='qe', mode='stack_orbitals',atoms=[1],elimit=[-13,6],plot_total=True)