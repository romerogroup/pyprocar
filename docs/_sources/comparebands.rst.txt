Compare bands
=============

This module is useful to compare different bands from different materials on the same band plot. The bands are plotted for the same :math:`k`-path in order to have a meaningful comparison but they do not need to have the same number of :math:`k`-points in each interval. The ``bandscompare()`` function contains all the parameters that are used in the ``bandsplot()`` along with an added feature of displaying a ``legend`` to help differentiate between the two different band structures. Different ``marker`` styles can be used as well. 

Usage::

	pyprocar.bandscompare('PROCAR1','PROCAR2',outcar='OUTCAR1',outcar2='OUTCAR2',cmap='jet',mode='parametric',marker='*',marker2='-.',elimit=[-5,5],kpointsfile='KPOINTS',legend='PRO1',legend2='PRO2',spin=1,spin2=2)

A similar approch could be used for other modes of band plots. 	

.. automodule:: pyprocar.scriptCompareBands
	:members: