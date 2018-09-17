import re
import os
import glob



def mergeabinit(outfile):
	#merge PROCAR files
	filenames = sorted(glob.glob("PROCAR_*"))

	with open(outfile, 'w') as outfile:
		for fname in filenames:
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)
