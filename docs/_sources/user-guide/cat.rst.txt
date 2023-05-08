.. _cat:

Concatenating multiple calculations
===================================

Multiple PROCAR files from multiple DFT calculations can be combined with this utility. This utility is particularly useful in cases of large systems, where one can split the bandstructure calculations along different high-symmetry directions in BZ, and then concatenate the PROCAR files for each separate :math:`k`-paths, and finally plot the full bandstructure in a single plot. The following command concatenates the PROCAR files obtained from three separate bandstructure calculations done along :math:`\Gamma-K`, :math:`K-M`, and :math:`M-\Gamma` :math:`k`-path in hexagonal Brillouin zone. 

Usage::

	pyprocar.cat(inFiles=['PROCAR_G-K','PROCAR_K-M','PROCAR_M-G'], outFile='PROCAR_merged', gz=False, mergeparallel = False, fixformat = False)

If the PROCARs are in a compressed .gz file, set ``gz=True``. If inFiles is not provided it will put all the ``PROCAR_*`` files into the inFiles list. 

NOTE:

When running Abinit in parallel the PROCAR is split into multiple files. To merge these files, set ``mergeparallel=True``. To detect if the calculation is spin polarized, provide ``abinit_output`` or set ``nspin`` value manually.
To fix formatting errors in Abinit PROCARs (spin directions not seperated, total projections not calculated) set ``fixformat=True`` as well. 

.. automodule:: pyprocar.scripts.scriptCat
	:members: