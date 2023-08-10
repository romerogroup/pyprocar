#!/usr/bin/env python3

import pyprocar.pyposcar as pp
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("inputfile", type=str, help="input file")
  parser.add_argument('--size', '-n', type=int, help="size in nearest neighbors"
                      " of the cluster to build", default=0)
  parser.add_argument('--smooth', '-s', action='store_true')
  parser.add_argument('--hydrogenate', '-y', action='store_true')

  parser.add_argument('-v', '--verbose', action='store_true')
  
  
  args = parser.parse_args()
  
  p = pp.poscar.Poscar(args.inputfile, verbose=False)
  p.parse()
  
  Defects = pp.defects.FindDefect(poscar=p,verbose=args.verbose)
  print(Defects.defects)

  # going to write a new file to mark the defects
  Defects.write_defects(method='any', filename=args.inputfile+'_defect.vasp')

  cluster = pp.clusters.Clusters(p, verbose=args.verbose,
                              neighbors=Defects.neighbors,
                              marked=Defects.all_defects)
  # just to avoid a warning in the automated test
  cluster.disable_warning = True
  cluster.extend_clusters(args.size)
  if args.smooth:
    cluster.smooth_edges()

  if args.hydrogenate:
    cluster.hydrogenate(args.inputfile+'_cluster.vasp')
  else:
    cluster.write(args.inputfile+'_cluster.vasp')

