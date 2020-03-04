import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import sys

class ProcarPlotCompare:
  def __init__(self, bands, bands2, spd, spd2, kpoints=None, kpoints2=None):
    self.bands = bands.transpose()
    self.spd = spd.transpose()
    self.kpoints = kpoints
    
    self.bands2 = bands2.transpose()
    self.spd2 = spd2.transpose()
    self.kpoints2 = kpoints2
    return

  def plotBands(self, size=0.02, size2=0.02, marker=',', marker2=',',color='r',color2='g', legend1='PROCAR1',legend2='PROCAR2', ticks=None,discontinuities=[]):
    if size is not None:
      size = size/2
    if size2 is not None:
      size2 = size2/2
      
    ### DATASET 1 ###

    if self.kpoints is not None:
      xaxis = [0]

      #### MODIFIED FOR DISCONTINOUS BANDS ####
      if ticks:
        ticks, ticksNames = list(zip(*ticks))  
        
        #counters for number of discontinuities 
        icounter = 1
        ii = 0
                             
        for i in range(1,len(self.kpoints)-len(discontinuities)):
          d = self.kpoints[icounter] - self.kpoints[icounter-1]
          d = np.sqrt(np.dot(d,d))
          xaxis.append(d + xaxis[-1])
          icounter+=1
          ii+=1
          if ii in discontinuities:
             icounter +=1
             ii+=1
             xaxis.append(xaxis[-1])
        xaxis = np.array(xaxis)  
         
        # plotting
        for i_tick in range(len(ticks)-1):
          x = xaxis[ticks[i_tick]:ticks[i_tick+1]+1]
          y = self.bands.transpose()[ticks[i_tick]:ticks[i_tick+1]+1,:] 
          plot = plt.plot(x,y,'r-', marker=marker,markersize=size,color=color)

      #### END  OF MODIFIED DISCONTINUOUS BANDS #### 
      

      # if ticks are not given
      else:
        xaxis = [0]
        for i in range(1,len(self.kpoints)):
          d = self.kpoints[i-1]-self.kpoints[i]
          d = np.sqrt(np.dot(d,d))
          xaxis.append(d+xaxis[-1])
        xaxis = np.array(xaxis) 

    # self.kpoints is None
    else: 
      xaxis = np.arange(len(self.bands))     
      plot = plt.plot(xaxis,self.bands.transpose() ,marker=marker,markersize=size,color=color) 

    ### DATASET 2 ###

    if self.kpoints2 is not None:
      xaxis2 = [0]

      #### MODIFIED FOR DISCONTINOUS BANDS ####
      if ticks:
                
        #counters for number of discontinuities 
        icounter = 1
        ii = 0
                             
        for i in range(1,len(self.kpoints2)-len(discontinuities)):
          d = self.kpoints2[icounter] - self.kpoints2[icounter-1]
          d = np.sqrt(np.dot(d,d))
          xaxis2.append(d + xaxis2[-1])
          icounter+=1
          ii+=1
          if ii in discontinuities:
             icounter +=1
             ii+=1
             xaxis2.append(xaxis2[-1])
        xaxis2 = np.array(xaxis2)  
         
        # plotting
        for i_tick in range(len(ticks)-1):
          x = xaxis2[ticks[i_tick]:ticks[i_tick+1]+1]
          y = self.bands2.transpose()[ticks[i_tick]:ticks[i_tick+1]+1,:] 
          plot = plt.plot(x,y,'r-', marker=marker2,markersize=size2,color=color2)

      #### END  OF MODIFIED DISCONTINUOUS BANDS #### 

      # if ticks are not given
      else:
        xaxis2 = [0]
        for i in range(1,len(self.kpoints2)):
          d = self.kpoints2[i-1]-self.kpoints2[i]
          d = np.sqrt(np.dot(d,d))
          xaxis2.append(d+xaxis2[-1])
        xaxis2 = np.array(xaxis2) 

    # self.kpoints is None
    else: 
      xaxis2 = np.arange(len(self.bands2))     
      plot = plt.plot(xaxis2,self.bands2.transpose() ,marker=marker2,markersize=size2,color=color2)
   
    ### Common parameters ### 

    #Pedram's workaround for legend
    plt.plot(np.ones((2,))*1e5000,np.ones((2,))*1e5000,label=legend1,color=color,marker=marker)
    plt.plot(np.ones((2,))*1e5000,np.ones((2,))*1e5000,label=legend2,color=color2,marker=marker2)
    plt.legend()    
    plt.xlim(min(xaxis.min(),xaxis2.min()), max(xaxis.max(),xaxis2.max()) )
    
    #handling ticks
    if ticks:
      #added for meta-GGA calculations
      if ticks[0] > 0:
        plt.xlim(left=xaxis[ticks[0]])
      if len(xaxis) > len(xaxis2):          
          ticks = [xaxis[x] for x in ticks]
      else:
          ticks = [xaxis2[x] for x in ticks]
      plt.xticks(ticks, ticksNames,fontsize=22)
      for xc in ticks:
           plt.axvline(x=xc,color='k')
    plt.yticks(fontsize=22)
    plt.axhline(color='r',linestyle='--')
    
    return plot

  def scatterPlot(self, size=50,size2=50, mask=None, cmap='jet' ,cmap2='hot_r', vmax=None, vmin=None,vmax2=None, vmin2=None,marker='o',marker2='o',legend1='PRO1',legend2='PRO2', ticks=None, discontinuities = []):
    bsize, ksize = self.bands.shape
    bsize2,ksize2 = self.bands2.shape
    print(bsize, ksize, bsize2, ksize2)

    ### DATASET 1 ###

    if self.kpoints is not None:
      xaxis = [0]

      #### MODIFIED FOR DISCONTINOUS BANDS ####  
      if ticks:
          ticks, ticksNames = list(zip(*ticks))  
      
          #counters for number of discontinuities 
          icounter = 1
          ii = 0
                               
          for i in range(1,len(self.kpoints)-len(discontinuities)):
            d = self.kpoints[icounter] - self.kpoints[icounter-1]
            d = np.sqrt(np.dot(d,d))
            xaxis.append(d + xaxis[-1])
            icounter+=1
            ii+=1
            if ii in discontinuities:
               icounter +=1
               ii+=1
               xaxis.append(xaxis[-1])
          xaxis = np.array(xaxis)  
          
          #plotting
          xaxis.shape=(1,ksize)
          xaxis = xaxis.repeat(bsize, axis=0)
          if mask is not None:
            mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
          else:
            mbands = self.bands
    
          plot = plt.scatter(xaxis, mbands, c=self.spd, s=size, linewidths=0,
                       cmap=cmap, vmax=vmax, vmin=vmin, marker=marker,
                       edgecolors='none')
        
      #### END  OF MODIFIED DISCONTINUOUS BANDS ####

      # if ticks are not given
      else:
        for i in range(1,len(self.kpoints)):
          d = self.kpoints[i-1]-self.kpoints[i]
          d = np.sqrt(np.dot(d,d))
          xaxis.append(d+xaxis[-1])
        xaxis = np.array(xaxis)    

        xaxis.shape=(1,ksize)
        xaxis = xaxis.repeat(bsize, axis=0)
        if mask is not None:
          mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
        else:
          mbands = self.bands
    
        plot = plt.scatter(xaxis, mbands, c=self.spd, s=size, linewidths=0,
                       cmap=cmap, vmax=vmax, vmin=vmin, marker=marker,
                       edgecolors='none')

    # if kpoints is None
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
    

    ### DATASET 2 ###

    if self.kpoints2 is not None:
      xaxis2 = [0]

      #### MODIFIED FOR DISCONTINOUS BANDS ####  
      if ticks:
         
      
          #counters for number of discontinuities 
          icounter = 1
          ii = 0
                               
          for i in range(1,len(self.kpoints2)-len(discontinuities)):
            d = self.kpoints2[icounter] - self.kpoints2[icounter-1]
            d = np.sqrt(np.dot(d,d))
            xaxis2.append(d + xaxis2[-1])
            icounter+=1
            ii+=1
            if ii in discontinuities:
               icounter +=1
               ii+=1
               xaxis2.append(xaxis2[-1])
          xaxis2 = np.array(xaxis2)  
          
          #plotting
          xaxis2.shape=(1,ksize2)
          xaxis2 = xaxis2.repeat(bsize2, axis=0)
          if mask is not None:
            mbands2 = np.ma.masked_array(self.bands2, np.abs(self.spd2) < mask)
          else:
            mbands2 = self.bands2
    
          plot = plt.scatter(xaxis2, mbands2, c=self.spd2, s=size2, linewidths=0,
                       cmap=cmap2, vmax=vmax2, vmin=vmin2, marker=marker2,
                       edgecolors='none')
        
      #### END  OF MODIFIED DISCONTINUOUS BANDS ####

      # if ticks are not given
      else:
        for i in range(1,len(self.kpoints2)):
          d = self.kpoints2[i-1]-self.kpoints2[i]
          d = np.sqrt(np.dot(d,d))
          xaxis2.append(d+xaxis2[-1])
        xaxis2 = np.array(xaxis2)    

        xaxis2.shape=(1,ksize2)
        xaxis2 = xaxis2.repeat(bsize2, axis=0)
        if mask is not None:
          mbands2 = np.ma.masked_array(self.bands2, np.abs(self.spd2) < mask)
        else:
          mbands2 = self.bands2
    
        plot = plt.scatter(xaxis2, mbands2, c=self.spd2, s=size2, linewidths=0,
                       cmap=cmap2, vmax=vmax2, vmin=vmin2, marker=marker2,
                       edgecolors='none')

    # if kpoints is None
    else:
      xaxis2 = np.arange(ksize2)
      xaxis2.shape=(1,ksize2)
      xaxis2 = xaxis2.repeat(bsize2, axis=0)
      if mask is not None:
        mbands2 = np.ma.masked_array(self.bands2, np.abs(self.spd2) < mask)
      else:
        mbands2 = self.bands2
    
      plot = plt.scatter(xaxis2, mbands2, c=self.spd2, s=size2, linewidths=0,
                       cmap=cmap2, vmax=vmax2, vmin=vmin2, marker=marker2,
                       edgecolors='none')
    
    # Common parameters

    plt.colorbar()
    plt.xlim(min(xaxis.min(),xaxis2.min()), max(xaxis.max(),xaxis2.max())) 
    
    #adding legend
    plt.plot(np.ones((2,))*1e2000,np.ones((2,))*1e2000,label=legend1,marker=marker)
    plt.plot(np.ones((2,))*1e2000,np.ones((2,))*1e2000,label=legend2,marker=marker2)
    plt.legend()  
    
    #handling ticks
    if ticks:
      #added for meta-GGA calculations
      if ticks[0] > 0:
        plt.xlim(left=xaxis[0,ticks[0]])
      if len(xaxis) > len(xaxis2): 
          ticks = [xaxis[0,x] for x in ticks]
      else:
          ticks = [xaxis2[0,x] for x in ticks]
      plt.xticks(ticks, ticksNames,fontsize=22)
    plt.yticks(fontsize=22)
    plt.axhline(color='r',linestyle='--')

    return plot
    
  def parametricPlot(self, cmap='jet' ,cmap2='hot_r', vmin=None, vmax=None,vmin2=None, vmax2=None ,mask=None, marker='--', marker2='-.', legend1='PROCAR1',legend2='PROCAR2',ticks=None, discontinuities = []):
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
    if vmin2 is None:
      vmin2 = self.spd2.min()
    if vmax2 is None:
      vmax2 = self.spd2.max()  
    print("normalizing to: ", (min(vmin,vmin2),max(vmax,vmax2)))
    norm = matplotlib.colors.Normalize(min(vmin,vmin2), max(vmax,vmax2))
    
    ### DATASET 1 ###

    #generating x axis data
    if self.kpoints is not None:
      xaxis = [0]

      ##### MODIFIED FOR DISCONTINUOUS BANDS #####

      if ticks:
          ticks, ticksNames = list(zip(*ticks))  
      
          #counters for number of discontinuities 
          icounter = 1
          ii = 0
                               
          for i in range(1,len(self.kpoints)-len(discontinuities)):
            d = self.kpoints[icounter] - self.kpoints[icounter-1]
            d = np.sqrt(np.dot(d,d))
            xaxis.append(d + xaxis[-1])
            icounter+=1
            ii+=1
            if ii in discontinuities:
               icounter +=1
               ii+=1
               xaxis.append(xaxis[-1])
          xaxis = np.array(xaxis)  
          
          #plotting
          for y,z in zip(mbands,self.spd):
            points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm, alpha=0.8,linestyles=marker)
            lc.set_array(z)
            lc.set_linewidth(1)
            gca.add_collection(lc)
      
      #### END  OF MODIFIED DISCONTINUOUS BANDS ####   
      
      # if ticks are not given
      else:
          for i in range(1,len(self.kpoints)):
            d = self.kpoints[i-1]-self.kpoints[i]
            d = np.sqrt(np.dot(d,d))
            xaxis.append(d+xaxis[-1])
          xaxis = np.array(xaxis) 
          
          #plotting
          for y,z in zip(mbands,self.spd):
            points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm, alpha=0.8,linestyles=marker)
            lc.set_array(z)
            lc.set_linewidth(1)
            gca.add_collection(lc)
              
    # if self.kpoints is None
    else:
        xaxis = np.arange(ksize)        
        for y,z in zip(mbands,self.spd):
           points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
           segments = np.concatenate([points[:-1], points[1:]], axis=1)
           lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm, alpha=0.8,linestyles=marker)
           lc.set_array(z)
           lc.set_linewidth(1)
           gca.add_collection(lc)

    ### DATASET 2 ###        

    #generating x axis data
    if self.kpoints2 is not None:
      xaxis2 = [0]

      ##### MODIFIED FOR DISCONTINUOUS BANDS #####

      if ticks:
          
      
          #counters for number of discontinuities 
          icounter = 1
          ii = 0
                               
          for i in range(1,len(self.kpoints2)-len(discontinuities)):
            d = self.kpoints2[icounter] - self.kpoints2[icounter-1]
            d = np.sqrt(np.dot(d,d))
            xaxis2.append(d + xaxis2[-1])
            icounter+=1
            ii+=1
            if ii in discontinuities:
               icounter +=1
               ii+=1
               xaxis2.append(xaxis2[-1])
          xaxis2 = np.array(xaxis2)  
          
          #plotting
          for y2,z2 in zip(mbands2,self.spd2):
            points2 = np.array([xaxis2, y2]).T.reshape(-1, 1, 2)
            segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
            lc2 = LineCollection(segments2, cmap=plt.get_cmap(cmap2), norm=norm, alpha=0.8,linestyles=marker2)
            lc2.set_array(z2)
            lc2.set_linewidth(1)
            gca.add_collection(lc2)
      
      #### END  OF MODIFIED DISCONTINUOUS BANDS ####   
      
      # if ticks are not given
      else:
          for i in range(1,len(self.kpoints2)):
            d = self.kpoints2[i-1]-self.kpoints2[i]
            d = np.sqrt(np.dot(d,d))
            xaxis2.append(d+xaxis2[-1])
          xaxis2 = np.array(xaxis2) 
          
          #plotting
          for y2,z2 in zip(mbands2,self.spd2):
            points2 = np.array([xaxis2, y2]).T.reshape(-1, 1, 2)
            segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
            lc2 = LineCollection(segments2, cmap=plt.get_cmap(cmap2), norm=norm, alpha=0.8,linestyles=marker2)
            lc2.set_array(z2)
            lc2.set_linewidth(1)
            gca.add_collection(lc2)
              
    # if self.kpoints is None
    else:
        xaxis2 = np.arange(ksize2)        
        for y2,z2 in zip(mbands2,self.spd2):
           points2 = np.array([xaxis2, y2]).T.reshape(-1, 1, 2)
           segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
           lc2 = LineCollection(segments2, cmap=plt.get_cmap(cmap2), norm=norm, alpha=0.8,linestyles=marker2)
           lc2.set_array(z2)
           lc2.set_linewidth(1)
           gca.add_collection(lc2)


    # Common parameters
    cb = plt.colorbar(lc)
    cb.ax.tick_params(labelsize=20)
    plt.legend((lc,lc2),(legend1,legend2))
    plt.xlim(min(xaxis.min(),xaxis2.min()), max(xaxis.max(),xaxis2.max()))  
    plt.ylim(min(mbands.min(),mbands2.min()), max(mbands.max(),mbands2.max()))

    #handling ticks
    if ticks:
      #added for meta-GGA calculations
      if ticks[0] > 0:
        plt.xlim(left=xaxis[ticks[0]])
      if len(xaxis) > len(xaxis2):          
          ticks = [xaxis[x] for x in ticks]
      else:
          ticks = [xaxis2[x] for x in ticks]
      plt.xticks(ticks, ticksNames,fontsize=22)
      for xc in ticks:
           plt.axvline(x=xc,color='lightgrey')
    plt.yticks(fontsize=22)
    plt.axhline(color='r',linestyle='--')

    return fig

  def atomicPlot(self, cmap='jet', cmap2='hot_r', vmin=None, vmax=None,vmin2=None, vmax2=None):
    """
    Just a handler to parametricPlot. Useful to plot energy levels. 

    It adds a fake k-point. Shouldn't be invoked with more than one
    k-point
    """

    print("Atomic plot: bands.shape  :", self.bands.shape)
    print("Atomic plot: spd.shape    :", self.spd.shape)
    print("Atomic plot: kpoints.shape:", self.kpoints.shape)
    
    print("Atomic plot: bands.shape #2 :", self.bands2.shape)
    print("Atomic plot: spd.shape  #2  :", self.spd2.shape)
    print("Atomic plot: kpoints.shape #2:", self.kpoints2.shape)

    self.bands = np.hstack((self.bands, self.bands))
    self.spd = np.hstack((self.spd, self.spd))
    self.kpoints = np.vstack((self.kpoints, self.kpoints))
    self.kpoints[0][-1] += 1
    
    self.bands2 = np.hstack((self.bands2, self.bands2))
    self.spd2 = np.hstack((self.spd2, self.spd2))
    self.kpoints2 = np.vstack((self.kpoints2, self.kpoints2))
    self.kpoints2[0][-1] += 1
    print("Atomic plot: bands.shape  :", self.bands.shape)
    print("Atomic plot: spd.shape    :", self.spd.shape)
    print("Atomic plot: kpoints.shape:", self.kpoints.shape)
    
    print("Atomic plot: bands.shape #2 :", self.bands2.shape)
    print("Atomic plot: spd.shape #2   :", self.spd2.shape)
    print("Atomic plot: kpoints.shape: #2 ", self.kpoints2.shape)

    print(self.kpoints, self.kpoints)
    
    fig = self.parametricPlot(cmap, min(vmin,vmin2), max(vmax,vmax2))
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # labels on each band
    for i in range(len(self.bands[:,0])):
      # print i, self.bands[i]
      plt.text(0, self.bands[i,0], str(i+1),color='r',fontsize=22)
      
    for i2 in range(len(self.bands2[:,0])):
      # print i, self.bands[i]
      plt.text(0, self.bands2[i2,0], str(i2+1),color='g',fontsize=22)  
    
    return fig