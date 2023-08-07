#!/usr/bin/env python3
import re 
import poscar
import argparse
import defects
import os

class plotBands:
    def __init__(self, POSCAR, PROCAR = "PROCAR", OUTCAR = "OUTCAR", savefigure = "savefigure"):
            self.poscar = POSCAR
            self.procar = PROCAR
            self.outcar = OUTCAR
            self.savefigure = savefigure
            #Some default values 
            self.ORB_Dict =  {"s" :  [0], "p":[1,2,3], "d":[4,5,6,7,8], "f":[9,10,11,12,13,14,15]}
            self.elimit = [-2,2]
            self.mode_ = "parametric"

    def plot(self, orbitals):
        
        """ Writes a script that uses pyprocar.bansplot() to plot the bands from a PROCAR and OUTCAR file
        for defect atoms from a given POSCAR file

        -Can select orbitals to plot
        -Can select a file to save the figure plotted
        """
        #Get list of defects for atoms
        f = open("plot_file.py", "w")
        f.write("#!/usr/bin/env python3 \n")
        f.write("import pyprocar \n")
        f.write("##Asumed default values are \n")
        f.write("#Energy limit values \n")
        f.write("elimit = [-2,2] \n")
        f.write("#Orbital dictionary used \n")
        f.write("ORB_Dict  = {'s' :  [0], 'p':[1,2,3], 'd':[4,5,6,7,8], 'f':[9,10,11,12,13,14,15]}\n")
        f.write("#POSCAR,PROCAR, OUTCAR, and Ploted figure files \n")
        f.write("POSCAR = '"+self.poscar+"'\n")
        f.write("PROCAR = '"+self.procar+"'\n")
        f.write("OUTCAR = '"+self.outcar+"'\n")
        f.write("savefigure = '"+self.savefigure+"'\n")




        POSCAR_ = poscar.Poscar(self.poscar)
        POSCAR_.parse()
        # looking for geometry features
        try:
          Defects = defects.FindDefect(POSCAR_)
          atoms  = Defects.all_defects
        # if there are none it will fail
        except IndexError:
          atoms = []
        #If no defects were found, default to use all atoms instead
        if(atoms == []):
            print("No defects were found on ", self.poscar)
            atoms = range(POSCAR_.Ntotal)
        line = "#The defects found on "+self.poscar+" were \n"
        f.write(line)
        f.write("atoms = [".rstrip("\n"))
        f.write(str(atoms[0]).rstrip("\n"))
        for i in atoms[1:]:
            f.write(",".rstrip("\n"))
            f.write(str(i).rstrip("\n"))
        f.write("] \n")


        asked_orb = []
        #Picking and writing Orbitals
        for i in orbitals:
            asked_orb += self.ORB_Dict[i]
        f.write("#Orbitals used for this plot\n")
        f.write("orbitals = [".rstrip("\n"))
        f.write(str(asked_orb[0]).rstrip("\n"))
        if(asked_orb[1:]):
            for i in asked_orb[1:]:
                f.write(",".rstrip("\n"))
                f.write(str(i).rstrip("\n"))
        f.write("] \n")

        #Checking OUTCAR for SPIN and Kpoint information
        out = open(self.outcar ,"r")
        out = out.readlines()
        if(out):
            match_KPOINTS = [re.findall(r'NKPTS\s*=\s*(\d*)', line) for line in out]
            match_SPIN = [re.findall(r'ISPIN\s*=\s*(\d*)', line) for line in out]
            KPOINTS = int([x for x in match_KPOINTS if(x)][0][0])
            ISPIN = int([x for x in match_SPIN if(x)][0][0])
            if(ISPIN == 2):
                f.write("cmap = 'seismic'\n")
                f.write("vmax = 1.0\n")
                f.write("vmin = -1.0\n")
            else:
                f.write("cmap = 'jet'\n")
                f.write("vmax = None\n")
                f.write("vmin = None\n")            
            if(KPOINTS != 1):
                f.write("mode = 'parametric'\n")
            else:
                f.write("mode = 'atomic'\n")
        else:
            print("Couldn't find OUTCAR file, assuming default mode and cmap")
            f.write("cmap = 'jet'\n")
            f.write("mode = 'parametric'\n")

        f.write("#We then plot the bands specified \n")
        f.write("pyprocar.bandsplot(PROCAR,outcar = OUTCAR,elimit = elimit,mode = mode,savefig = savefigure,atoms = atoms,orbitals = orbitals, cmap = cmap, vmax = vmax, vmin = vmin)")
        f.write('\n')
        f.close()
        os.system("chmod +rwx plot_file.py") 
        #pyprocar.bandsplot(PROCAR,outcar = outcar, elimit = elimit, mode = mode_,savefig = savefigure ,atoms = atoms, orbitals = asked_orb)

        return

if __name__ == '__main__':
    #plot(orbitals='s', POSCAR='POSCAR')
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbitals","-o", type=str,default = "spd", help = "Orbitals asked for ploting Ex = spd")
    parser.add_argument("--poscar","-g", type=str,default = "POSCAR", help = "POSCAR file name")
    parser.add_argument("--procar","-p", type=str,default = "PROCAR", help = "PROCAR file name")
    parser.add_argument("--outcar","-t", type=str,default = "OUTCAR", help = "OUTCAR file name")
    parser.add_argument("--bandplot","-b", type=str,default = "Band_Plot", help = "Plot file name")
    parser.add_argument("--run", action='store_true')
    args = parser.parse_args()
    if args.orbitals:
        orbitals = [*args.orbitals]
    if args.poscar:
        POSCAR = args.poscar
    if args.procar:
        PROCAR = args.procar
    if args.outcar:
        OUTCAR = args.outcar
    if args.bandplot:
        savefigure = args.bandplot

    
    plotBands(POSCAR = POSCAR, PROCAR = PROCAR, OUTCAR = OUTCAR, savefigure = savefigure).plot(orbitals = orbitals)
    if args.run:
        os.system("./plot_file.py")
        
        