Compare bands
=============

This module is useful to compare different bands from different materials on the same band plot. The bands are plotted for the same :math:`k`-path in order to have a meaningful comparison but they do not need to have the same number of :math:`k`-points in each interval. The ``bandscompare()`` function contains all the parameters that are used in the ``bandsplot()`` along with an added feature of displaying a ``legend`` to help differentiate between the two different band structures. Different ``marker`` styles can be used as well. 

Usage::

	pyprocar.bandscompare('PROCAR1','PROCAR2',outcar='OUTCAR1',outcar2='OUTCAR2',cmap='jet',mode='parametric',marker='*',marker2='-.',elimit=[-5,5],kpointsfile='KPOINTS',legend='PRO1',legend2='PRO2',spin=1,spin2=2, 
	kdirect=True, kdirect2=True)

A similar approch could be used for other modes of band plots. 	



=========================================
Export plot as a matplotlib.pyplot object
=========================================

PyProcar allows the plot to be exported as a matplotlib.pyplot object. This allows for further processing of the plot through options available in matplotlib.
This can be enabled by setting ``exportplt = True``.
Usage::

    import matplotlib.pyplot as plt
    import pyprocar

    plt = pyprocar.bandscompare('PROCAR1', 'PROCAR2', outcar='OUTCAR1', outcar2='OUTCAR2', mode='plain', exportplt=True)  
    plt.title('Using matplotlib options')
    plt.show()

.. automodule:: pyprocar.scriptCompareBands
	:members: