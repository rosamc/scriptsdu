import sys,os
import numpy as np
import matplotlib.pyplot as plt

def return_fullpars(pars,model,transitions,verbose=False):
    #pars shuold be in this order: kb,ku, forward basal parameters, reverse basal parameters, rates the TF changes
    #transitions shuld be [xf, yr]: TF modifies transition x in the forward direction, and y in the reverse. Starts at 1.
    sites,rev=model.split("_")
    nsites=int(sites)
    nrev=len(rev)
    kb,ku=pars[0:2]
    basal=pars[2:2+nsites] #forward basal parameters
    basalreverse=pars[2+nsites:2+nsites+nrev]
    #print(basalreverse)
    n=len(transitions)
    rates_TF=pars[-n:] #fold change increase or decrease when the TF is bound of each of the transitions
    pars_dict=dict()
    pars_dict["kb"]=kb
    pars_dict["ku"]=ku
    if model=="3_1":
        k_1_0,k_2_0,k_3_0=basal 
        kr_1_0=basalreverse[0]
        pars_dict["k_1_0"]=k_1_0
        pars_dict["k_2_0"]=k_2_0
        pars_dict["k_3_0"]=k_3_0
        pars_dict["kr_1_0"]=kr_1_0
        allparnames="kb,k_1_0,kr_1_0,kb,k_2_0,kb,k_3_0,ku,k_1_1,kr_1_1,ku,k_2_1,ku,k_3_1".split(",")
    elif model=="3_2":
        k_1_0,k_2_0,k_3_0=basal
        kr_2_0=basalreverse[0]
        pars_dict["k_1_0"]=k_1_0
        pars_dict["k_2_0"]=k_2_0
        pars_dict["k_3_0"]=k_3_0
        pars_dict["kr_2_0"]=kr_2_0
        allparnames="kb,k_1_0,kb,k_2_0,kr_2_0,kb,k_3_0,ku,k_1_1,ku,k_2_1,kr_2_1,ku,k_3_1".split(",")
    elif model=="3_12":
        k_1_0,k_2_0,k_3_0=basal
        kr_1_0,kr_2_0=basalreverse
        pars_dict["k_1_0"]=k_1_0
        pars_dict["k_2_0"]=k_2_0
        pars_dict["k_3_0"]=k_3_0
        pars_dict["kr_1_0"]=kr_1_0
        pars_dict["kr_2_0"]=kr_2_0
        allparnames="kb,k_1_0,kr_1_0,kb,k_2_0,kr_2_0,kb,k_3_0,ku,k_1_1,kr_1_1,ku,k_2_1,kr_2_1,ku,k_3_1".split(",")
    elif model=="5_1":
        k_1_0,k_2_0,k_3_0,k_4_0,k_5_0=basal 
        kr_1_0=basalreverse[0]
        pars_dict["k_1_0"]=k_1_0
        pars_dict["k_2_0"]=k_2_0
        pars_dict["k_3_0"]=k_3_0
        pars_dict["k_4_0"]=k_4_0
        pars_dict["k_5_0"]=k_5_0
        pars_dict["kr_1_0"]=kr_1_0
        allparnames="kb,k_1_0,kr_1_0,kb,k_2_0,kb,k_3_0,kb,k_4_0,kb,k_5_0,ku,k_1_1,kr_1_1,ku,k_2_1,ku,k_3_1,ku,k_4_1,ku,k_5_1".split(",")
    elif model=="5_1234":
        k_1_0,k_2_0,k_3_0,k_4_0,k_5_0=basal 
        kr_1_0,kr_2_0,kr_3_0,kr_4_0=basalreverse
        pars_dict["k_1_0"]=k_1_0
        pars_dict["k_2_0"]=k_2_0
        pars_dict["k_3_0"]=k_3_0
        pars_dict["k_4_0"]=k_4_0
        pars_dict["k_5_0"]=k_5_0
        pars_dict["kr_1_0"]=kr_1_0
        pars_dict["kr_2_0"]=kr_2_0
        pars_dict["kr_3_0"]=kr_3_0
        pars_dict["kr_4_0"]=kr_4_0
        allparnames="kb,k_1_0,kr_1_0,kb,k_2_0,kr_2_0,kb,k_3_0,kr_3_0,kb,k_4_0,kr_4_0,kb,k_5_0,ku,k_1_1,kr_1_1,ku,k_2_1,kr_2_1,ku,k_3_1,kr_3_1,ku,k_4_1,kr_4_1,ku,k_5_1".split(",")
    
    labels=[]
    for transition in transitions:
        if "f" in transition:
            label="k_%s_1"%transition[0]
        else:
            label="kr_%s_1"%transition[0]
        labels.append(label)
    
    
    fullpars=[]
    for parname in allparnames:
        if parname in pars_dict.keys(): #basal parameter, kb,ku
            fullpars.append(pars_dict[parname])
        else:#TF-bound parameter
            if parname in labels: #it is a transition modulated by the TF
                idx=labels.index(parname)
                basalpar=parname[0:-1]+"0" #replace 1 by 0 at the end to get the corresponding basal parameter
                #print(parname,basalpar)
                #sys.stdout.flush()
                fullpars.append(rates_TF[idx]*pars_dict[basalpar])
            else:
                basalpar=parname[0:-1]+"0" #replace 1 by 0 at the end
                #print(parname,basalpar)
                fullpars.append(pars_dict[basalpar])
    #print(allparnames)
    #print(fullpars)
    if verbose:
        print(list(zip(allparnames,fullpars)))
        
    return np.array(fullpars)

def get_constraints_npars_ccode(model,direction,fcd=100,fcu=100,tolerance=1e-6):
    #tolerance: to ensure TF always accelerates or decreases
    tol_p1=1+tolerance #a bit above 1
    tol_m1=1-tolerance #a bit below 1
    if model=="3_1":
        ccode=state3_rev1
    elif model=="3_2":
        ccode=state3_rev2
    elif model=="3_12":
        ccode=state3_rev12
    elif model=="5_1":
        ccode=state5_rev1
    elif model=="5_1234":
        ccode=state5_rev1234
    else:
        print("model not understood")
        raise(ValueError)

    if model=="3_1" or model=="3_2":
        npars=8
        p1=6
        p2=7
    elif model=="3_12":
        npars=9
        p1=7
        p2=8
    elif model=="5_1":
        npars=10
        p1=8
        p2=9
    elif model=="5_1234":
        npars=13
        p1=11
        p2=12
    else:
        print("model not understood")
        raise(ValueError)

    if direction=="p_m":#TF accelerates first transition it affects, reduces second
        constraints={p1:{'min':tol_p1,'max':fcu},p2:{'min':fcd,'max':tol_m1}}
    elif direction=="m_p": #reduces first, accelerates second
        constraints={p1:{'min':fcd,'max':tol_m1},p2:{'min':tol_p1,'max':fcu}}
    elif direction=="m_m":
        constraints={p1:{'min':fcd,'max':tol_m1},p2:{'min':fcd,'max':tol_m1}}
    elif direction=="p_p":
        constraints={p1:{'min':tol_p1,'max':fcu},p2:{'min':tol_p1,'max':fcu}}
    else:
        print("unrecognised direction, ", direction)
        raise ValueError
    return [constraints,npars,ccode]

def get_score_up_down(out):
    """monotonics, bell up, bell down (cup)"""
    
    if np.abs(np.max(out)-np.min(out))<0.001: #if it is essentially flat or the fold change is very small, then inaccuracies start appearing,and also it is not interesting...
        return([np.NaN,np.NaN])
    argmax=np.argmax(out)
    argmin=np.argmin(out)
    n=len(out)-1
    x0=out[0]
    x2=out[-1]
    if argmin==0 and argmax==n:#monotonically increasing
        x1=x2
    elif argmax==0 and argmin==n: #monotonically decreasing
        x1=x2
    else:
        if argmax>0 and argmax<n: #up and down
            x1=out[argmax]
        else:
            print("potentially down and up")
            print(",".join(map(str,out)))
            sys.stdout.flush()
            x1=out[argmin]
    delta_x=(x1-x0)/x0 #if increasing at x0: positive quantity. If decreasing at x0: negative.
    delta_y=(x2-x1)/x1 #if decreasing at x1: negative quantity. If increasing at x1: positive.
    return [delta_x,delta_y]

def score(pars,model=None,transitions=None,ccode=None,scoref=None,n=20,Amin=0,Amax=0,plot=False,returnout=False):
    
    fullpars=return_fullpars(pars,model,transitions)
    Avals=np.logspace(Amin,Amax,n)
    #print(len(Avals))
    out0=ccode.interfacess(fullpars,np.array([0])) #basal expression, in the absence of TF
    out=np.zeros(len(Avals))
    for a in range(n):
        A=Avals[a]
        out[a]=ccode.interfacess(fullpars,np.array([A]))
    score=scoref(out/out0)
    if plot:
        plt.plot(np.log10(Avals),out/out0)
        plt.show()
    if returnout:
        return [score,out/out0]
    else:
        return score