import numpy as np
from ..utils import welcome

def generate2dkmesh(x1:float, y1:float, x2:float, y2:float, z:float, nkx:int, nky:int,):
    """_summary_

    Parameters
    ----------
    x1 : float
        The beginning value to generate x coords
    y1 : float
        The beginning value to generate y coords
    x2 : float
        The end value to generate x coords
    y2 : float
        The end value to generate y coords
    z : float
        The z coordinate
    nkx : int
        The number of points to generate in the x direction
    nky : int
        The number of points to generate in the y direction

    Returns
    -------
    _type_
        _description_
    """
    
    welcome()

    kx = np.linspace(x1, x2, nkx)
    ky = np.linspace(y1, y2, nky)

    with open("Kgrid.dat", "w") as wf:
        wf.write("Generated by PyProcar\n")
        wf.write("%d\n" % (nkx * nky))
        wf.write("Reciprocal\n")

        kpoints = []
        for ikx in kx:
            for iky in ky:
                wf.write(
                    " {: >12.7f}   {: >12.7f}   {: >12.7f}   {: >12.7f}\n".format(
                        ikx, iky, z, 1.0
                    )
                )
                kpoints.append([ikx,iky,z])

    kpoints=np.array(kpoints)
    return kpoints
