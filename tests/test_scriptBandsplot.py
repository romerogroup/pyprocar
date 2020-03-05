"""
Testing bandplotting
"""

import pyprocar


pyprocar.bandsplot(
    file = './examples/SrVO3/nospin/PROCAR3',
    mode = 'parametric',
    elimit = [-6,6],
    orbitals = [4,5,6,7,8],
    vmin = 0,
    vmax = 1, 
    #code = 'elk',
    kpointsfile = './examples/SrVO3/nospin/KPOINTS3',
    outcar = './examples/SrVO3/nospin/OUTCAR3',
    savefig = './tests/test.pdf')
    
