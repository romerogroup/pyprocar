import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import sys

class ProcarPlot:
  def __init__(self, bands,bands2, spd,spd2, kpoints=None,kpoints2=None):
    self.bands = bands.transpose()
    self.spd = spd.transpose()
    self.kpoints = kpoints
    
    self.bands2 = bands2.transpose()
    self.spd2 = spd2.transpose()
    self.kpoints2 = kpoints2
    return

  def plotBands(self, size=None, marker='o', ticks=None):
    if size is not None:
      size = size/2
      
    if self.kpoints is not None:
      xaxis = [0]
      print "kpoints=",self.kpoints.shape
      for i in range(1,len(self.kpoints)):
        d = self.kpoints[i-1]-self.kpoints[i]
        d = np.sqrt(np.dot(d,d))
        xaxis.append(d+xaxis[-1])
      xaxis = np.array(xaxis)
    else:
      xaxis = np.arange(len(self.bands))
      
    #repeat for 2nd dataset
    if self.kpoints2 is not None:
      xaxis2 = [0]
      for i2 in range(1,len(self.kpoints2)):
        d2 = self.kpoints2[i2-1]-self.kpoints2[i2]
        d2 = np.sqrt(np.dot(d2,d2))
        xaxis2.append(d2+xaxis2[-1])
      xaxis2 = np.array(xaxis2)
    else:
      xaxis2 = np.arange(len(self.bands2))
    
    
    print "self.kpoints: ", self.kpoints.shape
    print "xaxis.shape : ", xaxis.shape
    print "bands.shape : ", self.bands.shape
    
    print "self.kpoints #2: ", self.kpoints2.shape
    print "xaxis.shape #2: ", xaxis2.shape
    print "bands.shape #2 : ", self.bands2.shape
    
    plot = plt.plot(xaxis,self.bands.transpose(), 'g',xaxis2,self.bands2.transpose(), 'r', marker=marker,markersize=size)
    plt.xlim(xaxis.min(), xaxis.max())

    #handling ticks
    if ticks:
      ticks, ticksNames = zip(*ticks)
      ticks = [xaxis[x] for x in ticks]
      plt.xticks(ticks, ticksNames)
    
    return plot
#
#  def scatterPlot(self, size=50, mask=None, cmap='hot_r', vmax=None, vmin=None,
#                  marker='o', ticks=None):
#    bsize, ksize = self.bands.shape
#    print bsize, ksize
#
#    if self.kpoints is not None:
#      xaxis = [0]
#      for i in range(1,len(self.kpoints)):
#        d = self.kpoints[i-1]-self.kpoints[i]
#        d = np.sqrt(np.dot(d,d))
#        xaxis.append(d+xaxis[-1])
#      xaxis = np.array(xaxis)
#    else:
#      xaxis = np.arange(ksize)
#
#    xaxis.shape=(1,ksize)
#    xaxis = xaxis.repeat(bsize, axis=0)
#    if mask is not None:
#      mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
#    else:
#      mbands = self.bands
#    
#    plot = plt.scatter(xaxis, mbands, c=self.spd, s=size, linewidths=0,
#                       cmap=cmap, vmax=vmax, vmin=vmin, marker=marker,
#                       edgecolors='none')
#    plt.colorbar()
#    plt.xlim(xaxis.min(), xaxis.max())
#    
#    #handling ticks
#    if ticks:
#      ticks, ticksNames = zip(*ticks)
#      ticks = [xaxis[0,x] for x in ticks]
#      plt.xticks(ticks, ticksNames)
#
#    return plot
    
  def parametricPlot(self, cmap='hot_r', vmin=None, vmax=None, mask=None, 
                     ticks=None):
    from matplotlib.collections import LineCollection
    import matplotlib
    fig = plt.figure()
    gca = fig.gca()
    bsize, ksize = self.bands.shape
    bsize2,ksize2=self.bands2.shape

    #print self.bands
    if mask is not None:
      mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
    else:
      #Faking a mask, all elemtnet are included
      mbands = np.ma.masked_array(self.bands, False)
    #print mbands
    
     #print self.bands for data set #2
    if mask is not None:
      mbands2 = np.ma.masked_array(self.bands2, np.abs(self.spd2) < mask)
    else:
      #Faking a mask, all elemtnet are included
      mbands2 = np.ma.masked_array(self.bands2, False)
    #print mbands
    
    #setting up vmin and vmax
    if vmin is None:
      vmin = self.spd.min()
    if vmax is None:
      vmax = self.spd.max()
    print "normalizing to: ", (vmin,vmax)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    
    #generating x axis data
    if self.kpoints is not None:
      xaxis = [0]
      for i in range(1,len(self.kpoints)):
        d = self.kpoints[i-1]-self.kpoints[i]
        d = np.sqrt(np.dot(d,d))
        xaxis.append(d+xaxis[-1])
      xaxis = np.array(xaxis)
    else:
      xaxis = np.arange(ksize)
      
    #generating x axis data for data set #2
    if self.kpoints2 is not None:
      xaxis2 = [0]
      for i2 in range(1,len(self.kpoints2)):
        d2 = self.kpoints2[i2-1]-self.kpoints2[i2]
        d2 = np.sqrt(np.dot(d2,d2))
        xaxis2.append(d2+xaxis2[-1])
      xaxis2 = np.array(xaxis2)
    else:
      xaxis2 = np.arange(ksize2)  

    #plotting
    for y,z in zip(mbands,self.spd):
      #print xaxis.shape, y.shape, z.shape
      points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
      segments = np.concatenate([points[:-1], points[1:]], axis=1)
      lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm, 
                          alpha=0.8)
      lc.set_array(z)
      lc.set_linewidth(2)
      gca.add_collection(lc)
    plt.colorbar(lc)
    plt.xlim(xaxis.min(), xaxis.max())
    plt.ylim(mbands.min(), mbands.max())
    
    
    #repeat for dataset #2
    for y2,z2 in zip(mbands2,self.spd2):
      #print xaxis.shape, y.shape, z.shape
      points2 = np.array([xaxis2, y2]).T.reshape(-1, 1, 2)
      segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
      lc2 = LineCollection(segments2, cmap=plt.get_cmap(cmap), norm=norm,alpha=0.8,linestyles='dotted')
      lc2.set_array(z2)
      lc2.set_linewidth(2)
      gca.add_collection(lc2)
      
      plt.legend((lc,lc2),('PROCAR1','PROCAR2'))
    

    #handling ticks
    if ticks:
      ticks, ticksNames = zip(*ticks)
      ticks = [xaxis[x] for x in ticks]
      plt.xticks(ticks, ticksNames)

    return fig

  def atomicPlot(self, cmap='hot_r', vmin=None, vmax=None):
    """
    Just a handler to parametricPlot. Useful to plot energy levels. 

    It adds a fake k-point. Shouldn't be invoked with more than one
    k-point
    """

    print "Atomic plot: bands.shape  :", self.bands.shape
    print "Atomic plot: spd.shape    :", self.spd.shape
    print "Atomic plot: kpoints.shape:", self.kpoints.shape

    self.bands = np.hstack((self.bands, self.bands))
    self.spd = np.hstack((self.spd, self.spd))
    self.kpoints = np.vstack((self.kpoints, self.kpoints))
    self.kpoints[0][-1] += 1
    print "Atomic plot: bands.shape  :", self.bands.shape
    print "Atomic plot: spd.shape    :", self.spd.shape
    print "Atomic plot: kpoints.shape:", self.kpoints.shape

    print self.kpoints
    
    fig = self.parametricPlot(cmap, vmin, vmax)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # labels on each band
    for i in range(len(self.bands[:,0])):
      # print i, self.bands[i]
      plt.text(0, self.bands[i,0], str(i+1), fontsize=15)
    
    return fig