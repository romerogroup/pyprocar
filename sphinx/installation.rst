Installation
============
PyProcar is supported by Python 3.x. 
Please install the following dependencies prior to installing PyProcar. 

	* matplotlib 
	* numpy 
	* scipy 
	* seekpath 
	* ase 
	* scikit-image
	* mayavi 
	* pychemia
	* pyvista

On Ubuntu, Fermi3D requires wxPython. Install it with::
    
    pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 wxPython

To install PyProcar with pip use the following command::
	
	pip install pyprocar

To install PyProcar with conda use the following command::
	
	conda install -c conda-forge pyprocar

Once you are done with the installation you can import PyProcar within Python with the command::

	import pyprocar

The command-line version of PyProcar can be invoked in a terminal with::
    
    procar	

Please note that the command-line version might not be up-to-date with the library version. 