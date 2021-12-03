import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def overlay_taylor_diagram_lines(axes,cax,option):
    '''
    Overlay lines emanating from origin on a Taylor diagram.

    OVERLAY_TAYLOR_DIAGRAM_CIRCLES(AXES,CAX,OPTION)
    Plots lines emanating from origin to indicate correlation values (CORs) 
 
    INPUTS:
    axes   : data structure containing axes information for target diagram
    cax    : handle for plot axes
    option : data structure containing option values. (Refer to 
             GET_TAYLOR_DIAGRAM_OPTIONS function for more information.)
    option['colcor']   : CORs grid and tick labels color (Default: blue)
    option['numberpanels'] : number of panels
    option['showlabelscor'] : Show or not the CORRELATION tick labels
    option['stylecor'] : Linestyle of the CORs grid
    option['tickcor']  : CORs values to plot lines from origin
    option['widthcor'] : Line width of the CORs grid
 
    OUTPUTS:
    None.

    Created on Dec 3, 2016
    Revised on Dec 3, 2021
    
    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    Modified by: Rafael Bischof
        Institute of Structural Engineering (IBK)
        https://ibk.ethz.ch/
        rabischof@ethz.ch
   '''

    # DRAW CORRELATION LINES EMANATING FROM THE ORIGIN:
    corr = option['tickcor'][option['numberpanels']-1]
    th  = np.arccos(corr)
    cst = np.cos(th); snt = np.sin(th);
    cs = np.append(-1.0*cst, cst)
    sn = np.append(-1.0*snt, snt)
    for i,val in enumerate(cs):
        plt.plot([0, axes['rmax']*cs[i]],[0, axes['rmax']*sn[i]], 
                 linestyle = option['stylecor'],
                 color = option['colcor'], linewidth = option['widthcor'])
    
    # annotate them in correlation coefficient
    if option['showlabelscor'] == 'on':
        fontSize = matplotlib.rcParams.get('font.size')
        rt = 1.05*axes['rmax']
        for i,cc in enumerate(corr):
            if option['numberpanels'] == 2:
                x = (1.05+abs(cst[i])/30)*axes['rmax']*cst[i]
            else:
                x = rt*cst[i]
            y = rt*snt[i]
            plt.text(x,y,str(round(cc,2)),
                     horizontalalignment = 'center',
                     color = option['colcor'], fontsize = fontSize)
