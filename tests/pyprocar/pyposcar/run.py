#!/usr/bin/env python3
import pyprocar.pyposcar as pp
import os


executable = '../../scripts/poscar.py'
auxdir = 'aux/'
resultsdir = 'results/'

POSCAR = ["POSCAR-7-9-B.vasp",
          "POSCAR-7-9.vasp" ,
          "POSCAR-C3Si.vasp",
          "POSCAR-C4.vasp" ,
          "POSCAR-C6H.vasp",
          "POSCAR-C6.vasp",
          "POSCAR-nv.vasp" ,
          "POSCAR-SiV.vasp",
          "POSCAR-V2.vasp"]

# identification of the defect
identification = {'options': ' ',
                  'suffix' : '_defect.vasp',
                  'outfile' : '_defect.vasp',
                  'print': '\nTesting the identification of defects (whole cell)'}
# clusters with #N nearest neighbors 
cluster_nn0 = {'options': ' -n 0 ',
               'suffix' : '_cluster.vasp',
               'outfile' : '_cluster.vasp',
               'print' : '\nTesting the building of clusters (0 nearest neighbors)'}
cluster_nn1 = {'options': ' -n 1 ',
               'suffix' : '_cluster-n1.vasp',
               'outfile' : '_cluster.vasp',
               'print' : '\nTesting the building of clusters (1 nearest neighbors)'}
cluster_nn2 = {'options': ' -n 2 ',
               'suffix' : '_cluster-n2.vasp',
               'outfile' : '_cluster.vasp',
               'print' : '\nTesting the building of clusters (2 nearest neighbors)'}
# smoothed clusters with #N nearest neighbors
s_cluster_nn1 = {'options': ' -n 1 -s ',
                 'suffix' : '_cluster-s1.vasp',
                 'outfile' : '_cluster.vasp',
                 'print' : '\nTesting smoothed  clusters (1 nearest neighbors)'}
s_cluster_nn2 = {'options': ' -n 2 -s ',
                 'suffix' : '_cluster-s2.vasp',
                 'outfile' : '_cluster.vasp',
                 'print' : '\nTesting smoothed  clusters (2 nearest neighbors)'}
# hydrogenated smoothed clusters with #N nearest neighbors
h_cluster_nn1 = {'options': ' -n 1 -s -y ',
                 'suffix' : '_cluster-h1.vasp',
                 'outfile' : '_cluster.vasp',
                 'print' : '\nTesting hydrogenated clusters (1 nearest neighbors)'}
h_cluster_nn2 = {'options': ' -n 2 -s -y ',
                 'suffix' : '_cluster-h2.vasp',
                 'outfile' : '_cluster.vasp',
                 'print' : '\nTesting hydrogenated clusters (2 nearest neighbors)'}

tasks = [identification,
         cluster_nn0,
         cluster_nn1,
         cluster_nn2,
         s_cluster_nn1,
         s_cluster_nn2,
         h_cluster_nn1,
         h_cluster_nn2]



for task in tasks:
  print(task['print'])
  for filename in POSCAR:
    
    # first running analize.py
    print(filename + ' ... ', end='')
    command = executable
    outfile = ' >> ' + auxdir + 'temp'
    cmdline = ' '.join([command, task['options'], filename, outfile])
    # print(cmd_1)
    os.system(cmdline)

    # second, moving the useful output to aux and removing the other file
    useful_str = filename + task['outfile']
    new_str = auxdir + filename + task['suffix']
    cmdline = 'mv ' + useful_str + ' ' + new_str
    os.system(cmdline)

    defect_str = filename  + '_defect.vasp' 
    cluster_str = filename + '_cluster.vasp'
    cmdline = " rm -f "+ defect_str + " " + cluster_str
    os.system(cmdline)
    

    # third, comparing the output with curated results
    path_p1 = auxdir + filename + task['suffix']
    path_p2 = resultsdir + filename + task['suffix']
    p1 = pp.poscar.Poscar(path_p1)
    p1.parse()
    p2 = pp.poscar.Poscar(path_p2)
    p2.parse()
    comparison = pp.poscarUtils.poscarDiff(p1, p2)

    if not comparison:
      print('ok')
    else:
      print('Results differs.')
      print(path_p1)
      print(path_p2)
      print(comparison)
      
    
#     print(poscarUtils.poscarDiff(poscar_defect_1,poscar_defect_2))
#   else:
#     continue
#   if(poscarUtils.poscarDiff(poscar_cluster_1,poscar_cluster_2)):
#     print(poscarUtils.poscarDiff(poscar_cluster_1,poscar_cluster_2))
#   else:
#     continue 
    





