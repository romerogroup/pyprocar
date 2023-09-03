from code import interact
import numpy as np
from . import latticeUtils
try:
  from sklearn.neighbors.kde import KernelDensity
except:
  from sklearn.neighbors import KernelDensity
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from . import generalUtils
from . import db


class RDF:
    #Add distances as optional argument
    def __init__(self, poscar = None):
        """This Class mainly obtains cutoff values for first neighbor
        criteria by utilizing KernelDensity It can obtain a single
        cutoff value for the whole distance matrix Or a cutoff value
        for each type of interaction (Ex = C-H).

        If there is no minimum of the KDE, I will take a very large
        value as cutoff.

        """
        self.distances = latticeUtils.distances(poscar.cpos,
                                                lattice=poscar.lat,
                                                allow_self=True)
        self.species = poscar.numberSp
        self.species_name = poscar.typeSp
        self.spDict = dict(zip(self.species_name,self.species))

        # print('self.spDict', self.spDict)
        # x-axis space for KDE calculations
        self.KDE_space = np.arange(0,6, 0.05)

        #Container for a single minimum value for all distances
        self.neighbor_threshold = None
        #Array of first minimums for each Interaction
        self.neighbor_thresholdSp = None
        #Matrix of first minimums formated to work with latticeUtils
        self.CutoffMatrix = None
        #All Species Interactions (C-H = H-C)
        self.interactions = None
        self.KDE_CurveSp()
        self.FindNeighborsSp()


    def _addUp(self,Species):
        """" Used to make some calculations easier
        returns the sum of all atom Species up until the species especified"""

        index = self.species_name.index(Species)
        add_up = int(np.sum(self.species[0:index]))

        return add_up
            
    def _findBlock(self, Interaction_X, Interaction_Y):
        """Finds the sub-matrix of interactions X-Y from the distances matrix """

        Starting_index_X = self._addUp(Species = Interaction_X)
        Starting_index_Y = self._addUp(Species = Interaction_Y)

        Block = self.distances[Starting_index_X : Starting_index_X + self.spDict[Interaction_X], Starting_index_Y:Starting_index_Y + self.spDict[Interaction_Y] ]

        return Block


    
        
    def KDE_CurveSp(self):
        """ Finding the kde curves for each interaction posible """
        kde_curveSp = []
        interactions = []
        for I in self.spDict.keys():
            for J in self.spDict.keys():
                
                #Here we skip the bottom half of the diagonal blocks in the distance matrix given that it is symmetric
                if(list(self.spDict.keys()).index(I) > list(self.spDict.keys()).index(J)):
                    continue

                aux_block = self._findBlock(Interaction_X = I,Interaction_Y = J)
                
                #We need to extract the zeros from the diagonals of the same Species interactions
                if(I == J):
                    aux_block = np.extract(1-np.eye(len(aux_block)), aux_block)
                else:
                    aux_block = aux_block.flatten()
                #Here we make sure that the sub_matrix taken contains physical distances
                #For example interactions C-C when there is only one C atom
                if (all(x == 0 for x in aux_block)):
                    continue
                aux_block = aux_block.reshape(-1,1)
                
                bandwidth = db.DB()
                #Here we get an adaptable Bandwidth that checks if there is H present
                #If there is bandwidth is 0.05, if there isn't, bandwidth is 0.1
                #This could be expanded into a whole database for each atom interaction or individualy
                bandwidth = bandwidth.get_bandwidth(I,J)
                kde_curve_fit = KernelDensity(kernel = 'gaussian', bandwidth=bandwidth).fit(aux_block)
                kde_curveSp.append(kde_curve_fit.score_samples(self.KDE_space.reshape(-1,1)))
                interactions.append([I,J])

        kde_curveSp = np.array(kde_curveSp)
        interactions = np.array(interactions)

        
        self.interactions = interactions
        #print('kde_curveSp', kde_curveSp)
      
        return kde_curveSp

    def KDE_Curve(self):

        """This calculates a single kde curve for the whole distance matrix"""
        
        non_zero_distances = np.extract(1-np.eye(len(self.distances)), self.distances)
        non_zero_distances = non_zero_distances.reshape(-1,1)

        kde_curve_fit = KernelDensity(kernel = 'gaussian', bandwidth=0.25).fit(non_zero_distances)

        kde_curve = kde_curve_fit.score_samples(self.KDE_space.reshape(-1,1))

        return kde_curve

    def FindNeighbors(self):

        """This method's main purpose is to find the single cutoff value for the whole distance matrix
         
        """

        kde = self.KDE_Curve()

        min_index = scipy.signal.argrelextrema(kde,np.less)[0]
        min = self.KDE_space[min_index][0]

        self.neighbor_threshold = min

        return min






    def FindNeighborsSp(self):

        """This method's main objective is to find the cutoff values for each Species
        As well as formating it as a matrix where the minimum [i,j] corresponds to the atom [i,j]
        in the distance matrix
        """
        mins = []
        KDE = self.KDE_CurveSp()
        neighborsSp = []
        temp_min_interaction = np.zeros(self.distances.shape)
        for kde, interaction in zip(KDE,self.interactions):
            # Here we check that the amount of data from the
            # interactions is enough to use Kernel Density If not, we
            # use the only distances available from the interaction
            data = self._findBlock(Interaction_X = interaction[0],Interaction_Y= interaction[1])
            if ((data.shape == (2,2))):
                min = data[0][1]*1.01
            elif((data.shape == (1,1))):
                min = data[0][0]*1.01
            else:
                # Here we make sure that there isn't any flat points
                # that could hide local minimums
                KDE_space, kde_aux = generalUtils.remove_flat_points(self.KDE_space , kde)

                min_index = scipy.signal.argrelextrema(kde_aux,np.less)[0]
                # If the method do not find any minimum, I will take the larger distance
                if len(min_index) == 0:
                  min_index = [-1]
                # print('min_index', min_index)
                  
                min = KDE_space[min_index][0]
                # print('min', min)
            mins.append(min)

            #Here we make the cutoff value matrix 
            x_start = self._addUp(Species=interaction[0])
            y_start = self._addUp(Species=interaction[1])
            temp_min_interaction[x_start : x_start + self.spDict[interaction[0]], y_start : y_start + self.spDict[interaction[1]] ] = min
            temp_min_interaction[y_start : y_start + self.spDict[interaction[1]], x_start : x_start + self.spDict[interaction[0]] ] = min

        self.CutoffMatrix = temp_min_interaction
        mins = np.array(mins)
        self.neighbor_thresholdSp = mins
         















