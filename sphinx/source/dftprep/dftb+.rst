.. _dftbplus: 

DFTB+ Preparation
==============================================

This guide is here to help you prepare DFTB+ Calculations to be used with pyprocar.

- Required files : bands.out, detailed.xml, detailed.out, eigenvectors.out
- flag           : code='dftb+'

In the DFTB+ code, the wavefunction is written in ``eigenvectors.out``
by setting ``WriteEigenvectors = Yes`` and ``EigenvectorsAsText =
Yes`` in the ``Analysis`` Block. From the wavefunction the projections
are calcualted by pyprocar.

The information of the energy levels (bands) are written in the file
``bands.out``, by setting ``WriteBandOut = Yes``, also in the
``Analisys`` block. The Fermi level, k-points, and crystal structure
is written in the files ``detailed.out`` and ``detailed.xml``. It
needs the flags ``WriteDetailedXML = Yes`` and ``WriteDetailedOut =
Yes`` in the ``Options`` block.

In summary, your ``dftb_in.hsd`` file should have the following sections:

.. code-block:: rst

    Analysis = {
    EigenvectorsAsText = Yes
    WriteEigenvectors = Yes
    WriteBandOut = Yes
    }

    Options = {
    WriteDetailedOut = Yes
    WriteDetailedXML = Yes
    }


*MIND*: Once pyprocar loads the DFTB+ wavefunctions, it creates a
 ``PROCAR`` file (vasp format) and other vasp-like ouput
 files. Loading a ``PROCAR`` is much faster than the DFTB+
 wavefunctions. If the ``PROCAR`` file is present, pyprocar
 automatically load it instead of the wavefunctions.

**Density of states is not supported**


Preparing Calculations
----------------------------------------------
To use DFTB+, one has to run various calculations in independent directories. Here, we will show examples for the different calculations.

Band Structure (SSC)
_______________________________________________
1. Create directory called ``scf``.
2. Perform self-consistent calculation in this ``scf`` directory.
3. Create directory called ``bands``.
4. Move the ``charges.bin`` file in the ``scf`` directory to the ``bands`` directory
5. Add the relevant k-points to the input ``dftb_in.hsd``. See an example in the `DFT recipes <https://dftbplus-recipes.readthedocs.io/en/latest/basics/bandstruct.html#calculating-the-band-structure>`
6. Make sure to add the tags ``WriteBandOut``, ``WriteDetailedXML``, ``WriteDetailedOut``, ``WriteEigenvectors``, ``EigenvectorsAsText`` to ``Yes`` in their respective blocks.
7. Perform a non-self consistent calculation, by adding ``ReadInitialCharges = Yes``, ``MaxSCCIterations = 1`` to the ``Hamiltonian`` block of ``dftb_in.hsd``
8. Run pyprocar.bandsplot(dirname = 'bands' ,mode = 'plain', code = 'dftb+')

Band Structure (non-SSC)
_______________________________________________
1. Add the relevant k-points to the input ``dftb_in.hsd``. See an example in the `DFT recipes <https://dftbplus-recipes.readthedocs.io/en/latest/basics/bandstruct.html#calculating-the-band-structure>`
2. Make sure to add the tags ``WriteBandOut``, ``WriteDetailedXML``, ``WriteDetailedOut``, ``WriteEigenvectors``, ``EigenvectorsAsText`` to ``Yes`` in their respective blocks.
3. Run pyprocar.bandsplot(dirname = 'bands' ,mode = 'plain', code = 'dftb+')

   

Fermi (SCC)
_______________________________________________
1. Create directory called ``scf``.
2. Perform self-consistent calculation in this ``scf`` directory.
3. Create directory called ``fermi``.
4. Move the ``charges.bin`` file in the ``scf`` directory to the ``fermi`` directory.
5. Make sure there is a kmesh in the ``KPointsAndWeights`` tag of the ``Hamiltonian`` block, ``dftb_in.hsd``  file in the ``fermi`` directory.
6. Make sure to add the tags ``WriteBandOut``, ``WriteDetailedXML``, ``WriteDetailedOut``, ``WriteEigenvectors``, ``EigenvectorsAsText`` to ``Yes`` in their respective blocks.
7. Perform a non-self consistent calculation in the ``fermi`` by setting ``ReadInitialCharges = Yes``, ``MaxSCCIterations = 1`` to the ``Hamiltonian`` block of ``dftb_in.hsd``.
8. Run pyprocar.FermiHandler(dirname = 'fermi', code = 'vasp')

Adding names to k-points
________________________
You can provide relevant names of the k-points as an argument to ``pyprocar.bandsplot(...)``.
Suppose you have the following ``Hamiltonian`` block ::

  Hamiltonian = DFTB {
    SCC = Yes
    SccTolerance = 1e-5
    MaxSCCIterations  = 1

    ...
    
    KPointsAndWeights =  KLines {
     1 0.0      0.0      0.0 
    10 0.333333 0.333333 0.0 
    10 0.5      0.0      0.0 
    10 0.0      0.0      0.0 
    }
  }

you can use pyprocar.bandsplot(dirname = 'bands', mode = 'plain', code = 'dftb+', kticks=[0,10,20,30], knames=[r'\Gamma', 'K', 'M', r'\Gamma'])
