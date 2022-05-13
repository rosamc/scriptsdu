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
    else:
        raise ValueError("unrecognised model", model)
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

def get_constraints_npars(model,direction,fcd=100,fcu=100,tolerance=1e-6):
    #tolerance: to ensure TF always accelerates or decreases
    tol_p1=1+tolerance #a bit above 1
    tol_m1=1-tolerance #a bit below 1
    

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
    return [constraints,npars]

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

def get_score_up_down_v2(out,tol=1e-5,check_out0=True,out0_tol=0.001,fc_tol=0.070389327891398):
    """out is assumed to be the response, in log2(FC).
    tol is the tolerance to decide if at any given point, the function is increasing, decreasing or flat. 
    out0_tol is used to discard functions if they start with a value greater than out0_tol. This is only applied if check_out0 is set to True.
    fc_tol is set to 0.07039 which is log2(1.05/1). If the difference between maximum and minimum of out is less than this, it is essentially flat, so discard."""
    if check_out0 and out[0]>out0_tol:
        #print(out[0],"evaluate a smaller TF concentration") 
        return([np.NaN, np.NaN])
    if np.abs(np.max(out)-np.min(out))<fc_tol: #if it is essentially flat or the fold change is very small, then inaccuracies start appearing,and also it is not interesting...
        return([np.NaN,np.NaN])    
    argmax=np.argmax(out)
    argmin=np.argmin(out)
    n=len(out)-1
    y0=out[0]
    y1=out[-1]
    dif=np.diff(out)
    up=np.where(dif>tol)[0]
    flat=np.where((dif>-tol)&(dif<tol))[0]
    down=np.where(dif<-tol)[0]
    mono_up=False
    mono_down=False
    up_down=False
    down_up=False
    multi_peak=True
    if len(down)==0:
        mono_up=True
        yc=y1
    elif len(up)==0:
        mono_down=True
        yc=y0
    elif np.max(up)<np.min(down): #increasing and decreasing: all up are less than all down
        up_down=True
        yc=out[argmax]
    elif np.max(down)<np.min(up): #decreasing and increasing: all down are less than all up
        down_up=True
        yc=out[argmin]
    else:
        multi_peak=True
        print(multi_peak)
        print(",".join(map(str,out)))
        #mins=argrelmin(out)
        #maxs=argrelmax(out)
        #plt.plot(range(len(out)),out)
        #plt.scatter(mins,out[mins],color="b")
        #plt.scatter(maxs,out[maxs],color="r")
        #plt.title("%s,%s"%(0,y1))
        #plt.show()
        yc=0
        
        
        
            
    delta_x=(yc) #if increasing at x0: positive quantity. If decreasing at x0: negative.
    delta_y=(y1) #if the last value is greater than the first: positive, otherwise: negative.  
    return [delta_x,delta_y]

def score(pars,model=None,transitions=None,ccode=None,scoref=None,n=20,Amin=0,Amax=0,plot=False,returnout=False,log2out=False,n_per_om=4,**kwargs):
    """kwargs are arguments to be passed to the actual scoring function scoref. It must have a fc_tol parameter to pick Amin and Amax in this function, and other parameters as needed for scoref."""
    print(kwargs)
    tol_A=kwargs["out0_tol"]#to decide Amax and Amin
    fullpars=return_fullpars(pars,model,transitions)
    out0=ccode.interfacess(fullpars,np.array([0])) #basal expression, in the absence of TF
    score=[]
    if not Amin:
        Amin=1e-3
        #first find Amin and Amax such that function starts at 0 and is saturated
        Amin_found=False
        i=0
        while ((Amin_found is False) and (i<20)):
            outA0=ccode.interfacess(fullpars,np.array([Amin]))
            if np.abs(np.log2(outA0/out0))<tol_A:
                Amin_found=True
                #print("when Amin is found", np.log2(outA0/out0))
            else:
                Amin=Amin/10
                i+=1
        if not Amin_found:
            #print("failed to get Amin",out0, outA0)
            score=[np.NaN,np.NaN]
    if not Amax:
        #print("searching Amax")
        Amax=1e3
        Amax_found=False
        outinf25=ccode.interfacess(fullpars,np.array([1e25])) #very saturated expression
        outinf30=ccode.interfacess(fullpars,np.array([1e30])) 
        if np.abs(np.log2(outinf30/outinf25))>tol_A:
            #print("not saturated at 30:", outinf25, outinf30)
            score=[np.NaN,np.NaN]
        
        if len(score)==0:
            i=0
            while ((Amax_found is False) and (i<20)):
                outAm=ccode.interfacess(fullpars,np.array([Amax]))
                if np.abs(np.log2(outAm/outinf30))<tol_A:
                    Amax_found=True
                else:
                    Amax=Amax*10
                    i+=1
            if not Amax_found:
                #print("failed to get Amax", outinf30, outAm)
                score=[np.NaN,np.NaN]
    if len(score)==0 or plot:
        if not n:
            log10Amin=np.log10(Amin)
            log10Amax=np.log10(Amax)
            orders_m=log10Amax-log10Amin
            n=int(n_per_om*np.ceil(orders_m))
        Avals=np.logspace(log10Amin,log10Amax,n)
    
        #print(len(Avals))
        out=np.zeros(len(Avals))
        for a in range(n):
            A=Avals[a]
            out[a]=ccode.interfacess(fullpars,np.array([A]))
        if not log2out:
            f=out/out0
        else:
            f=np.log2(out/out0)
        if len(score)==0:
            score=scoref(f,**kwargs)
    
    if plot:
        plt.plot(np.log10(Avals),f)
        plt.show()
    if returnout:
        return [score,f,Amin,Amax,n]
    else:
        return score