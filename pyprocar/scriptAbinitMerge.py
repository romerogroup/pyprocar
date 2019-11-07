import re
import os
import glob



def mergeabinit(outfile):
	""" This module merges PROCAR files generated from multiple 
	Abinit calculations.
	"""
	filenames = sorted(glob.glob("PROCAR_*"))

	with open(outfile, 'w') as outfile:
		for fname in filenames:
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)
