Band gap calculation
===================================

The bandgap of a calculation can be found by::

	pyprocar.bandgap(procar="PROCAR", outcar="OUTCAR", code="vasp")

For ``Abinit`` calculations set outcar to be the output file. For other DFT codes, setting only ``code`` is sufficient when run from the calculation directory. 

NOTE:
The bandgap calculation should be done for non-self consistent (band structure) calculations. 


.. automodule:: pyprocar.scriptBandGap
	:members: