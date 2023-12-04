# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:06:35 2023

@author: hbehrooz
"""
import numpy as np
#from scipy.optimize import least_squares
#import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import warnings
#conda install -c conda-forge imageio
import imageio.v2 as imageio
from os.path import isfile 
#remove created figures
files = glob.glob('fig/*')
for f in files:
    os.remove(f)
    
#np.random.seed(19680801)
# costants
#the status of  various device types
DSC=3 #discovered but not changed to BS cause by low accuracy
EMD=2 #device is an EMD
BS=1 # device is an MD that converted to a Base Station 
MD=0 #device is an mobile device

# R=250 # the radious of the impacted area
# N=1000 # number of impacted persons(mobile device) in the impacted area
EMD_range=100 #coverage range of EMDs
MD_range=[10,20,30,40,50] #list of various coverages for mobile devices 
N_EMD=4 #number of EMDs
RSS0_initial =31.7 # recivied signal strength at refrence distance=1m
accuracy=.05 # accuracy of localization in term of average prediction of distances between BSs and MD

Mute=False #define if the output of optimization will be mutely work

#number of independent varaibles x,y,v
N_independ=3

#center of the the scenario
centerX=0
centerY=0

#the x and y coordination of EMDs
EMDx=[-125,-100,-120,-85]
EMDy=[-25,25, 22,-10]



# Define the system of nonlinear equations as a function
    
def residual_val(x,xbs,RSS,RSS0):

    # Define your equations here
    F_val=np.zeros(len(xbs))
    #loop over the base stations
    for k in range(len(xbs)):
        F_val[k]= ((x[0]-xbs[k,0])**2+(x[1]-xbs[k,1])**2)**.5-(10.**((RSS[k]-RSS0[k])/(10.*x[2])))
    return F_val

#Jacobian matrix of the system of equations
def jac_mat(x,xbs,RSS,RSS0) : 
    J=np.zeros((len(xbs),len(x)))
    for k in range (len(xbs)):
        J[k,0]=(x[0]-xbs[k,0])/(((x[0]-xbs[k,0])**2+(x[1]-xbs[k,1])**2)**.5)
        J[k,1]=(x[1]-xbs[k,1])/(((x[0]-xbs[k,0])**2+(x[1]-xbs[k,1])**2)**.5)
        J[k,2]=(np.log(10)*((RSS[k]-RSS0[k])*10**(((RSS[k]-RSS0[k])/(10*x[2]))-1)))/(x[2]**2)
    return(J)  
 
#the Levenberg-Marquardt method implmentation
def LM(x0,xbs,RSS,RSS0,lambd=.001,epsilon=1e-16,progress=.0001,max_iter=1000, criteri_len=10):
    """
    

    Parameters
    ----------
    x0 : intila guess
        DESCRIPTION.
    xbs : coordiantion of base stations 
        DESCRIPTION.
    RSS : RSS values
        DESCRIPTION.
    RSS0 : RSS0 values
        DESCRIPTION.
    lambd : Tthe lambda value for algorithm
        DESCRIPTION. The default is 3.
    epsilon : an lower stoping criteria for cost tob used for stoping , optional
        DESCRIPTION. The default is 1e-16.
    progress : lower bound  of unprogressing cost for stoping criteri, optional
        DESCRIPTION. The default is .0001.
    max_iter : maximum iteration, optional
        DESCRIPTION. The default is 1000.
        
    criteri_len: # number of iteration with least change in results

    Returns
    -------
    None.

    """
    warnings.filterwarnings("error") #set to catch the runtime warning cause by matrix division by zero

    x_old=x0
    res =[] #is a list of residual during iterations
    x_l=[] # x value duirng iterations
    lambdL=[] #lambda values during iterations
    try:
        for iter in range(max_iter):
            
            A=jac_mat(x_old,xbs,RSS,RSS0)
            B=A.T@A
            C=B+lambd*np.diag(np.diag(B))
            E=residual_val(x_old,xbs,RSS,RSS0)
            D=-A.T@E
            try: # if the mtrix is uninversible (singular) use alteranative solution
                inv_C = np.linalg.inv(C)
            except:
                if(Mute!=False):
                    print("Singular matrix not inversible Moore-Penrose pseudo-inverse is used")
                inv_C = np.linalg.pinv(C)
                
            dx=inv_C@D
            x_new=x_old+dx
            E_new=residual_val(x_new,xbs,RSS,RSS0)
            Obj_f_new=(np.sum(E_new**2)**.5)/len(E_new)
            res.append(Obj_f_new)
            x_l.append(x_new)
            lambdL.append(lambd)
            if(iter!=0): #if itis not first iteration
                if(res[iter]<epsilon): #if the residual value is less than epsilon the solution is converged 
                    break            
                elif(res[iter]>res[iter-1]): #if current residual is greater than last one increase lambda by 11 times
                    lambd*=11
                else:
                    lambd/=9  #if current residual is less than last one decreas lambda by 9 times
                    x_old=x_new            
            else:
                x_old=x_new
    # if the the residual compare to last criteri_len iterations average is less than a progress value 
    # the progress is not change signicntly during last criteri_len itteration and then stop
            if(iter>criteri_len) & (abs(res[iter]-np.mean(res[iter-criteri_len:]))<epsilon):
                break
    except RuntimeWarning:
        return(np.nan,np.nan,np.nan)
    return(x_new,res,lambdL)
            
   

# creat a GIF file from figures in fig/ directory
def make_gif(des='figures.gif',duration=1000,loop=10):

    images = []
    for iteration in range(1,1000000):
        filename="fig/fig%d.png"%(iteration)
        if(isfile(filename)) :

           images.append(imageio.imread(filename))
        else:
           break
    imageio.mimsave(des, images,duration=1000,loop=10)
    
#plot the nodes colored by their status  
def plot_nodes(N,R,nodes,accuracy,iteration=-1,save=0,labeled=False):
    fig, ax = plt.subplots()
    tab_palet=['Undiscovered MD','Changed to BS','EMD','Discoveredc MD']
    color_palet=['tab:red', 'tab:green','tab:blue','tab:purple']
    nodes_discoverd=np.sum(nodes['cost']==nodes['cost'])
    nodes_discoverd_acc=len(nodes[(nodes['cost']==nodes['cost']) &(nodes['status']==BS)])
    for status in [MD,BS,EMD,DSC]:
        if(status==DSC):
            sub_nodes= nodes[(nodes['status']==MD)&(nodes['x_cal']==nodes['x_cal'])]
        else:
            sub_nodes= nodes[nodes['status']==status]
        if(labeled):
            
            ax.scatter(sub_nodes['x'], sub_nodes['y'], c=color_palet[status],label=tab_palet[status],
                           alpha=0.8, edgecolors='none')
        else:
                        
            ax.scatter(sub_nodes['x'], sub_nodes['y'], c=color_palet[status],
                           alpha=0.8, edgecolors='none')
    ax.legend(loc='upper right')
    if(iteration!=-1):
        ax.set_title("N=%d, R=%dm,acc=%f \nIteration:%3d [%3d discovered(Transformed=%3d)]"%(N,R,accuracy,iteration,nodes_discoverd,nodes_discoverd_acc),loc='left')
    ax.grid(True)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.show()
    if(save):
        ax.figure.savefig("fig/fig%d.png"%(iteration), dpi=300)
    plt.close(fig)
    
 


#distribute n device in a area with R radius with center of (centerX,centerY)
def device_distribution(n,R,centerX=0,centerY=0):
    #here explain why we have to use square root of random number
    #https://www.anderswallin.net/2009/05/uniform-random-points-in-a-circle-using-polar-coordinates/
    r = R*( np.random.uniform(size=n)**.5)
    theta =np.random.uniform(0, 2 * math.pi,size=n) 
    x = centerX + r * list(map(math.cos, theta))
    y = centerY + r * list(map(math.sin, theta))
    v = 2+np.random.default_rng().gamma(2, .4, n) # a value for power loss exponent
    RSS0=28+np.random.uniform(size=n)*7 # a random value for RSS0
    # x =np.random.uniform(-R, R,size=n)
    # y =np.random.uniform(-R, R,size=n)
    return(x,y,v,RSS0)

#the function get a distance d from a base station with its RSS0 and v values and change the disnace to a RSS value
# and add a noise_percent percent amount of white noise to it 
def rev_PL(d,v_val,RSS0_val,noise_percent=.05):
    rss=np.zeros(len(d))
    for k in range(len(d)):
        if(d[k]>0.01):# to be sure that the points are not overlaped
          rss[k]=(10 *np.log10(d[k])*v_val)+RSS0_val[k]
    if(noise_percent!=0):
        rss= rss*(1+np.random.normal(0,noise_percent))
    return(rss)

#simulate a envirnemt containd N mobile devices within R radious and consider an accuracy for 
#finding the coordination of the mobile device
def simulate(N,R,accuracy=.05):
    #hyper paramteres for LM function
    LM_precion=0.001 # the minimum accetable cost value for accepting a coordination from  LM function 
    lambd=0.01
    epsilon=1e-16 # 
    max_iter=1000 #maximum number of iteration
    
    
    noise_add=0.05#.001 # an amount of white noise added to RSS value 
    upper_limit=2.0 # an uppper limit for accuracy (residual ) returning by LM function. if any solution
    #accuraccy exceede from this  value it will not be considered for transformaing to a Base Station
    ############
    
    #creat a scenario with N number of device in an area of R radious
    x,y,v,RSS0=device_distribution(N,R)
    #this the dataframe cotians each nodes in the scenario
    nodes = pd.DataFrame(columns=['id' #id of mobile device
                                  ,'x','y'# orginal coordinations of mobile device aassigned by simualtion env 
                                  ,'range' #coverage range of mobile device 
                                  ,'status' #shows if the device is an EMD MD, or base station
                                  ,'dist' #distance which is each time calcuated for one specific MD to all other divices
                                  ,'x_cal','y_cal' #estimated coordination for MD
                                  ,'v' #v value for MD assigned by simulation Env
                                  ,'RSS0' #RSS0 value for MD assigned by simualtion Env.
                                  ])
    #loop over EMDs and put them on dataframe
    for k in range(N_EMD):
        nodes.loc[len(nodes.index)]=[len(nodes.index),EMDx[k],EMDy[k],EMD_range,EMD,np.nan,np.nan,np.nan,2,RSS0_initial]
    #loop over devices and put them on dataframe
    for k in range(N):
        nodes.loc[len(nodes.index)]=[len(nodes.index),x[k],y[k],MD_range[np.random.randint(len(MD_range))],MD,np.nan,np.nan,np.nan,v[k],RSS0[k]]
    nodes['cost']=np.nan #accuracy value returned by LM function for coordination
    nodes['N']=np.nan    #umber of discovered neighbors
    nodes['iteration']=np.nan #iteration which the coordination is established
    nodes['v_cal']=np.nan # v value which is estiamted
    #Show if we found new Base Station in last iteration
    New_BS_discoverd=1 # a flag that shows if a new base station is transfromed during last iteration
    Cost_val=[] 
    iteration=0  #iteration number
    nodes_changed=0 #nodes that had accuratly coordinated and changed to a BS

    
    while (New_BS_discoverd)  :  # while a new base station was transformed during last iteration
        iteration+=1
        New_BS_discoverd=0
        plot_nodes(N,R,nodes,accuracy,iteration,save=1,labeled=False)
    # looping over each exisiting  device
        for ind in list(nodes.index):
            #do somthing if the node is an MD
            if(nodes.loc[ind,'status']==MD): #use it if it is a MD not a base station or EMD
                #find the distance of currrent node form other nodes
                cx=nodes.loc[ind,'x']
                cy=nodes.loc[ind,'y']
                dist=((cx-nodes['x'])**2+ (cy-nodes['y'])**2)**.5 #calcualte distance of curren MD from all other divices.
                nodes['dist']=dist
                #transform the distance to an RSS  value from all other devices toward this MD
                RSS = rev_PL(nodes['dist'],nodes.loc[ind,'v'],nodes['RSS0'],noise_add)
                nodes['RSS']=RSS
                #select neighboring nodes that current MD is in their range of coverage and they are base station or EMD
                in_range_nodes=nodes[(dist<nodes['range']) & (nodes['status']!=MD) &(nodes.index!=ind)]
                #if we found more than 3 neighbors current MD could be changeed to a BS 
                if(len(in_range_nodes)>N_independ):
                    print("in_range",in_range_nodes)
                    print("ind",ind)
                    #it should call trilatation to estimate x,y here
                    xbs=in_range_nodes[['x','y']].values # coordination of the neighboring BS
                    RSS=in_range_nodes['RSS'].values
                    RSS0=in_range_nodes['RSS0'].values
                    #intialize a guess for independent variables by taking the average of in range BS (x,y,v)
                    initial_guess=in_range_nodes[['x','y','v']].mean()
    # use LM algorithm to estiamte x,y,v for the cuurent MD
                    x_pre,resid,lambdL=LM(initial_guess,xbs,RSS,RSS0,
                                         lambd=lambd,epsilon=epsilon,progress=LM_precion,max_iter=max_iter)
                    if(np.all(x_pre==x_pre) ): #if there was not any matrix calculation error
                        if((resid[-1]<upper_limit)): #if the cost function not passed the upper limit
                        # then this estiamtion could be assigned as a valid paramter estimation for current MD
                            nodes.loc[ind,'x_cal']=x_pre[0]
                            nodes.loc[ind,'y_cal']=x_pre[1]
                            nodes.loc[ind,'v_cal']=x_pre[2]
                            nodes.loc[ind,'iteration']=iteration
        
                            nodes.loc[ind,'cost']=resid[-1]
                            nodes.loc[ind,'N']=len(in_range_nodes)                        
                            if resid[-1]<accuracy: # if cost is below some level of accyracy
                            # if the accracy o estimation is acceptable then MD can transform to an BS
                                Cost_val.append(resid)
                                nodes_changed+=1
                                New_BS_discoverd=1
                                nodes.loc[ind,'status']=BS
                                if(Mute!=False):
                                    print("+++(MD/Cost)=",ind,round(resid[-1],4))
    
                    else:
                        if(Mute!=False):
                            print("---(MD/Cost)=",ind,round(resid[-1],4))

                    
    
                                           
    make_gif(des='N=%d_R=%d_acc=%3f_fig.gif'%(N,R,accuracy))  
    discoverd_nodes=np.sum(nodes['cost']==nodes['cost'])              
    print ("number of discovered nodes:%d"%(discoverd_nodes))
    print ("number of discovered nodes accurartly to be used as BS:%d"%(nodes_changed))
    
    print("Cost value statistical measures with noise added=%f:\n"%(noise_add),"\n",nodes['cost'].describe()  ) 
    #plot_nodes(nodes,save=1)
    nodes['err_cord']=((nodes.x-nodes.x_cal)**2+(nodes.y-nodes.y_cal)**2)**.5
    return(nodes,iteration)
#%%
#evaluate RMSE errors in independent varaibles estimations
def error(nodes,iterates,accuracy):
    nodes_c=nodes.dropna()
    RMSE_v =(np.sum((nodes_c.v-nodes_c.v_cal)**2)/len(nodes_c))**.5
    RMSE_x=(np.sum((nodes_c.x-nodes_c.x_cal)**2)/len(nodes_c))**.5
    RMSE_y=(np.sum((nodes_c.y-nodes_c.y_cal)**2)/len(nodes_c))**.5
    RMSE_t=(np.sum((nodes_c.v-nodes_c.v_cal)**2+(nodes_c.x-nodes_c.x_cal)**2+(nodes_c.y-nodes_c.y_cal)**2)/len(nodes_c))**.5
    N_MD=len(nodes[(nodes['x_cal']==nodes['x_cal']) &(nodes['status']==MD)])
    N_BS=len(nodes[(nodes['x_cal']==nodes['x_cal']) &(nodes['status']==BS)]) 
    print("iterations=%d,coordinated MD=%d(BS=%d)"%(iterates,N_MD,N_BS))
    print("with accuracy=",accuracy,"root Square errors are:")
    print("RMSE x=%f\nRMSE y=%f\nRMSE v=%f"%(RMSE_x,RMSE_y, RMSE_v))
    return(RMSE_t,RMSE_v,RMSE_x,RMSE_y,N_BS,N_MD,np.mean(nodes.cost))

#evaluate errors for various accuracy 

nodes_l=[]
iter_l=[]
acc_l=[]

for accuracy in [1.8,1.4,1.2,1,.8,.5,.25,.1,.05,.02,.01]:
    nod,ite=simulate(1000,250,accuracy)
    acc_l.append(accuracy)
    iter_l.append(ite)
    nodes_l.append(nod)
df = pd.DataFrame(columns=['Accuracy','x','y','v','total','#iteration','#BS','#MD','ave_cost'])
for k in range(len(nodes_l)) :
    print("-------------------------------")
    t,v1,x1,y1,bs,md,cost=error(nodes_l[k],iter_l[k],acc_l[k])
    df.loc[len(df.index)]=[acc_l[k],x1,y1,v1,t,iter_l[k],bs,md,cost]
df.to_pickle('static.pkl')
#%%
df=pd.read_pickle('static.pkl')

        
##################                  

