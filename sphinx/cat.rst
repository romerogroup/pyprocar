Concatenating multiple calculations
===================================

Multiple PROCAR files from multiple DFT calculations can be combined with this utility. This utility is particularly useful in cases of large systems, where one can split the bandstructure calculations along different high-symmetry directions in BZ, and then concatenate the PROCAR files for each separate :math:`k`-paths, and finally plot the full bandstructure in a single plot. The following command concatenates the PROCAR files obtained from three separate bandstructure calculations done along :math:`\Gamma-K`, :math:`K-M`, and :math:`M-\Gamma` :math:`k`-path in hexagonal Brillouin zone. 

Usage::

	pyprocar.cat(['PROCAR_G-K','PROCAR_K-M','PROCAR_M-G'],'PROCAR_merged')

.. automodule:: pyprocar.scriptCat
	:members:

To concatenate PROCAR's generated from Abinit assuming the files are all in the same directory, use the following command::

	pyprocar.mergeabinit('PROCAR_merged')

.. automodule:: pyprocar.scriptAbinitMerge
	:members: