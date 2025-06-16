.. raw:: html

    <div class="banner">
        <h2 style="text-align: center; color: #2E86AB; margin-bottom: 20px;">PyProcar: Pre- and Post-processing of Density Functional Theory Codes</h2>
        <p style="text-align: center; font-size: 18px; color: #555; margin-bottom: 30px;">
            A robust, open-source Python library for electronic structure analysis
        </p>
    </div>

.. toctree::
	:hidden:

	getting-started/index
	user-guide/index
	dftprep/index
	examples/index
	api/index

PyProcar is a robust, open-source Python library used for pre- and post-processing of the electronic structure data coming from DFT calculations. 
PyProcar provides a set of functions that manage data from the PROCAR format obtained from various DFT codes. 
Basically, the PROCAR file is a projection of the Kohn-Sham states over atomic orbitals. 
That projection is performed to every :math:`k`-point in the considered mesh, every energy band and every atom. 

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
        <h3 style="color: white; margin-bottom: 15px;">Ready to get started?</h3>
        <p style="color: #f0f0f0; margin-bottom: 20px;">Explore our comprehensive examples and user guide</p>
        <a href="./getting-started/index.html" style="background: white; color: #667eea; padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: bold; margin-right: 10px;">Get Started</a>
        <a href="./examples/index.html" style="background: rgba(255,255,255,0.2); color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: bold; border: 2px solid white;">View Examples</a>
    </div>



Showcase: What PyProcar Can Do
==============================

PyProcar is capable of performing a multitude of tasks including plotting plain and spin/atom/orbital projected band structures and 
Fermi surfaces- both in 2D and 3D, Fermi velocity plots, unfolding bands of a super cell, comparing band structures from multiple DFT calculations, 
plotting partial density of states and generating a :math:`k`-path for a given crystal structure.




Crystal Field Splitting in SrVO₃
=================================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/SrVO3_Crystal Field Splitting.png" alt="SrVO3 Crystal Field Splitting" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Orbital-projected band structure showing crystal field splitting of d-orbitals in SrVO₃, 
            illustrating how PyProcar can decompose electronic bands by orbital character to reveal crystal field effects in transition metal compounds.
        </p>
    </div>

Dirac Point Identification in Graphene
=======================================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/Graphene_Dirac Point.png" alt="Graphene Dirac Point" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Band structure of Graphene highlighting the Dirac points where conduction and valence bands meet, 
            showcasing PyProcar's capability to identify and analyze topological features and linear dispersion relations in 2D materials.
        </p>
    </div>

Fermi Surface Analysis - Gold Van Alphen Frequencies
=====================================================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/Gold_Van Alphen.png" alt="Gold Van Alphen Frequencies" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Fermi surface cross-sections and Van Alphen oscillation frequencies for Gold, 
            demonstrating PyProcar's ability to analyze the topology of Fermi surfaces and calculate quantum oscillation properties in metals.
        </p>
    </div>


Spin-Orbit Coupling and Rashba Effect in BiSb Monolayer
========================================================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/bisb_monolayer_Spin Rashba.PNG" alt="BiSb Monolayer Spin Rashba Effect" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Spin-resolved band structure of BiSb monolayer showing the Rashba spin-orbit coupling effect, 
            demonstrating PyProcar's ability to visualize spin-polarized electronic structures and analyze spin-orbit interactions in topological materials.
        </p>
    </div>

Band Structure Visualization
=============================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/bands_showcase.png" alt="Band Structure Showcase" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Electronic band structure showing energy bands along high-symmetry k-points, 
            demonstrating PyProcar's capability to generate clean, publication-ready band structure plots with customizable styling and projection options.
        </p>
    </div>

2D Band Structure Analysis
==========================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/bands2d_showcase.png" alt="2D Band Structure Showcase" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Two-dimensional band structure visualization showing energy dispersion across the entire Brillouin zone, 
            highlighting PyProcar's ability to create comprehensive 2D band maps for analyzing electronic properties and band topology.
        </p>
    </div>

Density of States Analysis
==========================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/dos_showcase.png" alt="Density of States Showcase" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Density of states (DOS) with orbital and species projections, 
            showcasing PyProcar's ability to decompose the total DOS into atomic and orbital contributions for detailed electronic structure analysis.
        </p>
    </div>

2D Fermi Surface Mapping
========================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/fermi2d_showcase.png" alt="2D Fermi Surface Showcase" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Two-dimensional Fermi surface cross-sections showing the topology of the Fermi level, 
            demonstrating PyProcar's capability to visualize Fermi surfaces in 2D planes with high resolution and customizable projections.
        </p>
    </div>

3D Fermi Surface Visualization
==============================

.. raw:: html

    <div style="text-align: center; margin: 40px 0; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa;">
        <img src="_static/images/fermi3d_showcase.png" alt="3D Fermi Surface Showcase" style="width: 80%; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
        <p style="margin-top: 20px; color: #333; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;">
            <strong>What is being plotted:</strong> Three-dimensional Fermi surface rendering showing the complete topology of electronic states at the Fermi level, 
            highlighting PyProcar's advanced 3D visualization capabilities for comprehensive Fermi surface analysis and interactive exploration.
        </p>
    </div>

Supported DFT Codes
===================

Currently supports:

1. **VASP** - Vienna Ab initio Simulation Package
2. **Quantum Espresso** - Plane-wave pseudopotential code
3. **Abinit** - Plane-wave pseudopotential code  
4. **Elk** - Full-potential linearized augmented-plane wave code (Band Structure and Fermi surface support in development)
5. **Lobster** - Chemical bonding analysis (Still in development)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


