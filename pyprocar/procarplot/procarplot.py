import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import sys

class ProcarPlot:
  def __init__(self, bands, spd, kpoints=None):
    self.bands = bands.transpose()
    self.spd = spd.transpose()
    self.kpoints = kpoints
    return

  def plotBands(self, size=0.02, marker='o', ticks=None,color='blue'):
    if size is not None:
      size = size/2
    if self.kpoints is not None:
      xaxis = [0]
      for i in range(1,len(self.kpoints)):
        d = self.kpoints[i-1]-self.kpoints[i]
        d = np.sqrt(np.dot(d,d))
        xaxis.append(d+xaxis[-1])
      xaxis = np.array(xaxis)
    else:
      xaxis = np.arange(len(self.bands))
    print("self.kpoints: ", self.kpoints.shape)
    print("xaxis.shape : ", xaxis.shape)
    print("bands.shape : ", self.bands.shape)
    plot = plt.plot(xaxis,self.bands.transpose(), 'r-', marker=marker, 
                    markersize=size,color=color)
    plt.xlim(xaxis.min(), xaxis.max())

    #handling ticks
    if ticks:
      ticks, ticksNames = list(zip(*ticks))
      #added for meta-GGA calculations
      if ticks[0] > 0:
      	plt.xlim(left=xaxis[ticks[0]])
      ticks = [xaxis[x] for x in ticks]
      plt.xticks(ticks, ticksNames,fontsize=22)
      plt.yticks(fontsize=22)
    
    return plot

  def scatterPlot(self, size=50, mask=None, cmap='hot_r', vmax=None, vmin=None,
                  marker='o', ticks=None):
    bsize, ksize = self.bands.shape
    print(bsize, ksize)

    if self.kpoints is not None:
      xaxis = [0]
      for i in range(1,len(self.kpoints)):
        d = self.kpoints[i-1]-self.kpoints[i]
        d = np.sqrt(np.dot(d,d))
        xaxis.append(d+xaxis[-1])
      xaxis = np.array(xaxis)
    else:
      xaxis = np.arange(ksize)

    xaxis.shape=(1,ksize)
    xaxis = xaxis.repeat(bsize, axis=0)
    if mask is not None:
      mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
    else:
      mbands = self.bands
    
    plot = plt.scatter(xaxis, mbands, c=self.spd, s=size, linewidths=0,
                       cmap=cmap, vmax=vmax, vmin=vmin, marker=marker,
                       edgecolors='none')
    plt.colorbar()
    plt.xlim(xaxis.min(), xaxis.max())
    
    #handling ticks
    if ticks:
      ticks, ticksNames = list(zip(*ticks))
      #added for meta-GGA calculations
      if ticks[0] > 0:
      	plt.xlim(left=xaxis[0,ticks[0]])
      ticks = [xaxis[0,x] for x in ticks]
      plt.xticks(ticks, ticksNames,fontsize=22)
      plt.yticks(fontsize=22)

    return plot
    
  def parametricPlot(self, cmap='jet', vmin=None, vmax=None, mask=None, 
                     ticks=None):
    from matplotlib.collections import LineCollection
    import matplotlib
    fig = plt.figure()
    gca = fig.gca()
    bsize, ksize = self.bands.shape

    #print self.bands
    if mask is not None:
      mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
    else:
      #Faking a mask, all elemtnet are included
      mbands = np.ma.masked_array(self.bands, False)
    #print mbands
    
    if vmin is None:
      vmin = self.spd.min()
    if vmax is None:
      vmax = self.spd.max()
    print("normalizing to: ", (vmin,vmax))
    norm = matplotlib.colors.Normalize(vmin, vmax)

    if self.kpoints is not None:
      xaxis = [0]
      for i in range(1,len(self.kpoints)):
        d = self.kpoints[i-1]-self.kpoints[i]
        d = np.sqrt(np.dot(d,d))
        xaxis.append(d+xaxis[-1])
      xaxis = np.array(xaxis)
    else:
      xaxis = np.arange(ksize)

    for y,z in zip(mbands,self.spd):
      #print xaxis.shape, y.shape, z.shape
      points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
      segments = np.concatenate([points[:-1], points[1:]], axis=1)
      lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm)
      lc.set_array(z)
      lc.set_linewidth(1)
      gca.add_collection(lc)
    cb = plt.colorbar(lc)
    cb.ax.tick_params(labelsize=20)
    plt.xlim(xaxis.min(), xaxis.max())
    plt.ylim(mbands.min(), mbands.max())

    #handling ticks
    if ticks:
      ticks, ticksNames = list(zip(*ticks))
      #added for meta-GGA calculations
      if ticks[0] > 0:
      	plt.xlim(left=xaxis[ticks[0]])
      ticks = [xaxis[x] for x in ticks]
      plt.xticks(ticks, ticksNames,fontsize=22)
      plt.yticks(fontsize=22)

    return fig

  def atomicPlot(self, cmap='hot_r', vmin=None, vmax=None):
    """
    Just a handler to parametricPlot. Useful to plot energy levels. 

    It adds a fake k-point. Shouldn't be invoked with more than one
    k-point
    """

    print("Atomic plot: bands.shape  :", self.bands.shape)
    print("Atomic plot: spd.shape    :", self.spd.shape)
    print("Atomic plot: kpoints.shape:", self.kpoints.shape)

    self.bands = np.hstack((self.bands, self.bands))
    self.spd = np.hstack((self.spd, self.spd))
    self.kpoints = np.vstack((self.kpoints, self.kpoints))
    self.kpoints[0][-1] += 1
    print("Atomic plot: bands.shape  :", self.bands.shape)
    print("Atomic plot: spd.shape    :", self.spd.shape)
    print("Atomic plot: kpoints.shape:", self.kpoints.shape)

    print(self.kpoints)
    
    fig = self.parametricPlot(cmap, vmin, vmax)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # labels on each band
    for i in range(len(self.bands[:,0])):
      # print i, self.bands[i]
      plt.text(0, self.bands[i,0], str(i+1), fontsize=22)
    
    return fig
