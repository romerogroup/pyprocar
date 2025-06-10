# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 23:54:16 2020

@author: lllan
"""

import re
from pathlib import Path
from typing import Union

import numpy as np


class FrmsfParser:
    def __init__(self, filepath: Union[str, Path] = Path("in.frmsf")):

        self.filepath = Path(filepath)

        rf = open(self.filepath)
        self.data = rf.readlines()
        rf.close()

        self.rec_lattice = None
        self.numBands = None
        self.origin = None
        self.numPoints = None
        self.bandLabels = None
        self.bandData = None

        self.bands = None
        self.parse_frmsf()

    def parse_frmsf(self):
        self.numPoints = np.array([int(x) for x in self.data[0].split()])
        self.kpointGenerationMethod = int(self.data[1])

        if self.kpointGenerationMethod == 0:

            def kGeneration(n1, n2, n3, N1, N2, N3):

                return np.array(
                    [
                        (2 * n1 - 1 - N1) / N1,
                        (2 * n2 - 1 - N2) / N2,
                        (2 * n3 - 1 - N3) / N3,
                    ]
                )

        elif self.kpointGenerationMethod == 1:

            def kGeneration(n1, n2, n3, N1, N2, N3):

                return np.array([(n1 - 1) / N1, (n2 - 1) / N2, (n3 - 1) / N3])

        elif self.kpointGenerationMethod == 2:

            def kGeneration(n1, n2, n3, N1, N2, N3):

                return np.array(
                    [
                        (2 * n1 - 1) / (2 * N1),
                        (2 * n2 - 1) / (2 * N2),
                        (2 * n3 - 1) / (2 * N3),
                    ]
                )

        self.numkpoints = self.numPoints[0] * self.numPoints[1] * self.numPoints[2]
        self.numBands = int(self.data[2])
        self.rec_lattice = np.array(
            [[float(y) for y in x.split()] for x in self.data[3:6]]
        )
        self.values = np.array([float(x) for x in self.data[6:]])
        # counter += 1
        self.numProjections = int(
            (len(self.values) - (self.numkpoints * self.numBands)) / (self.numkpoints)
        )
        self.bands = np.zeros(shape=[self.numkpoints, self.numBands])
        self.kpoints = np.zeros(shape=[self.numkpoints, 3])

        self.projections = np.zeros(
            shape=[int(self.numkpoints), int(self.numProjections)]
        )

        counter = 0
        for iproperty in range(1, 3):

            for iband in range(self.numBands):
                kpointCounter = 0
                for i in range(1, self.numPoints[0] + 1):
                    for j in range(1, self.numPoints[1] + 1):
                        for k in range(1, self.numPoints[2] + 1):
                            if iproperty == 1:
                                self.bands[kpointCounter, iband] = self.values[counter]
                                self.kpoints[kpointCounter, :] = kGeneration(
                                    n1=i,
                                    n2=j,
                                    n3=k,
                                    N1=self.numPoints[0],
                                    N2=self.numPoints[1],
                                    N3=self.numPoints[2],
                                )
                            elif iproperty == 2:
                                self.projections[kpointCounter, iband] = self.values[
                                    counter
                                ]

                            kpointCounter += 1
                            counter += 1
