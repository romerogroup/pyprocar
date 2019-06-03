.. _labelkmesh:

k-mesh generator
================

This utility can be used to generate a 2D :math:`k`-mesh centered at a given :math:`k`-point and in a given :math:`k`-plane. This is particularly useful in computing 2D spin-textures and plotting 2D Fermi-surfaces. For example, the following command creates a 2D :math:`k_{x}`-:math:`k_{y}` -mesh centered at the :math:`\Gamma` point (:math:`k_{z}= 0`) ranging from coordinates (-0.5, -0.5, 0.0) to (0.5, 0.5, 0.0) with a grid size of 0.02:

General format::  

	pyprocar.generate2dkmesh(x1,y1,x2,y2,grid_size)

Usage::

	pyprocar.generate2dkmesh(-0.5,-0.5,0.5,0.5,0.02)
