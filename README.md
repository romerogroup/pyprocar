[![PyPI version](https://badge.fury.io/py/pyprocar.svg)](https://badge.fury.io/py/pyprocar)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/pyprocar.svg?label=conda-forge&colorB=027FD5)](https://anaconda.org/conda-forge/pyprocar)
[![Build Status](https://travis-ci.org/romerogroup/pyprocar.svg?branch=master)](https://travis-ci.org/romerogroup/pyprocar)
[![HitCount](http://hits.dwyl.com/uthpalaherath/romerogroup/pyprocar.svg)](http://hits.dwyl.com/uthpalaherath/romerogroup/pyprocar)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyprocar)

Notice
===========
- **New Release** We recently updated to a new version 'v6.2.1'.
- **Support for Previous Versions**: For users who prefer to continue with an older version, we have conveniently archived the previous releases on GitHub, and provided a link to the corresponding documentation.


PyProcar
===========

PyProcar is a robust, open-source Python library used for pre- and post-processing of the electronic structure data coming from DFT calculations. PyProcar provides a set of functions that manage data obtained from the PROCAR format. Basically, the PROCAR format is a projection of the Kohn-Sham states over atomic orbitals. That projection is performed to every k-point in the considered mesh, every energy band and every atom. PyProcar is capable of performing a multitude of tasks including plotting plain and spin/atom/orbital projected band structures and Fermi surfaces- both in 2D and 3D, Fermi velocity plots, unfolding bands of a super  cell, comparing band structures from multiple DFT calculations, plotting partial density of states and generating a k-path for a given crystal structure.

Currently supports:

1. VASP
2. Elk (Stll in development)
3. Quantum Espresso
4. Abinit (DOS Stll in development)
5. Lobster (Stll in development)

![](welcome.png)

Documentation
-------------

For versions 6.1.0 and above, the documentation is found here:
https://romerogroup.github.io/pyprocar/


The prior documentation is found here:
https://romerogroup.github.io/pyprocar5.6.6/


Developers
------------
Francisco Muñoz <br />
Aldo Romero <br />
Sobhit Singh <br />
Uthpala Herath <br />
Pedram Tavadze <br />
Eric Bousquet <br />
Xu He <br />
Reese Boucher <br />
Logan Lang <br />
Freddy Farah <br />


How to cite
-----------
If you have used PyProcar in your work, please cite:

- U. Herath, P. Tavadze, X. He, E. Bousquet, S. Singh, F. Muñoz, and A. H. Romero, PyProcar: A Python library for electronic structure pre/post-processing, Computer Physics Communications 251, 107080 (2020). DOI: <https://doi.org/10.1016/j.cpc.2019.107080>

- L. Lang, P. Tavadze, A. Tellez, E. Bousquet, H. Xu, F. Muñoz, N. Vasquez, U. Herath, and A. H. Romero, Expanding PyProcar for new features, maintainability, and reliability, Computer Physics Communications 297, 109063 (2024). DOI: <https://doi.org/10.1016/j.cpc.2023.109063>

Thank you.

BibTex:

    @article{HERATH2020107080,
    title = "PyProcar: A Python library for electronic structure pre/post-processing",
    journal = "Computer Physics Communications",
    volume = "251",
    pages = "107080",
    year = "2020",
    issn = "0010-4655",
    doi = "https://doi.org/10.1016/j.cpc.2019.107080",
    url = "http://www.sciencedirect.com/science/article/pii/S0010465519303935",
    author = "Uthpala Herath and Pedram Tavadze and Xu He and Eric Bousquet and Sobhit Singh and Francisco Muñoz and Aldo H. Romero",
    keywords = "DFT, Bandstructure, Electronic properties, Fermi-surface, Spin texture, Python, Condensed matter",
    }

    @article{LANG2024109063,
    title = {Expanding PyProcar for new features, maintainability, and reliability},
    journal = {Computer Physics Communications},
    volume = {297},
    pages = {109063},
    year = {2024},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2023.109063},
    url = {https://www.sciencedirect.com/science/article/pii/S0010465523004083},
    author = {Logan Lang and Pedram Tavadze and Andres Tellez and Eric Bousquet and He Xu and Francisco Muñoz and Nicolas Vasquez and Uthpala Herath and Aldo H. Romero},
    keywords = {Electronic structure, DFT, Post-processing},
    }


Mailing list
-------------
Please post your questions on our forum.

https://groups.google.com/d/forum/pyprocar

Dependencies
------------
matplotlib <br />
numpy <br />
scipy <br />
seekpath <br />
ase <br />
scikit-image <br />
pychemia <br />
pyvista <br />

Installation
------------

with pip:

	pip install pyprocar

with conda:

    conda install -c conda-forge pyprocar

Usage
-----
Typical use is as follows

    import pyprocar
    pyprocar.bandsplot(code='vasp',mode='plain', dirname='bands')

Previously, bandsplot would accept the OUTCAR and PROCAR file paths as inputs,
in v6.0.0 we moved to specifying the directory where the bands calculation took place.

Refer to the documentation for further details.

Stand-alone mode:

    procar.py -h

will bring a help menu.

Changelog
--------------

For the old changelog, see [CHANGELOG.md](CHANGELOG.md)
