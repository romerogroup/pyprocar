Repair
======

This utility is used to repair the ill-formatting of the PROCAR file due to the erroneous file handling in Fortran, particularly in a VASP calculation. This prevents issues arising from the lack of white space between a number and a negative sign, for instance ``0.000000-0.5000000``. Typically, ``pyprocar.repair()`` is recommended to be applied before using any other utility. Setting ``permissive=True`` during plotting will achieve this too. 

Usage::

	pyprocar.repair(`PROCAR',`PROCAR-repaired')

.. automodule:: pyprocar.scriptRepair
	:members: