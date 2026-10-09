"""Parameter-count reference: two-input tanh MLP, analytic full-batch Adam."""
import time
import numpy as np
from scipy.special import expit
from .core import loss, probability
from .runtime import check


def width(target):
    candidates=[0]+list(range(1,max(2,(target+3)//4+2)))
    return min(candidates,key=lambda h:(abs((3 if h==0 else 4*h+1)-target),h))


class MLP:
    def __init__(self,h): self.h=h;self.d=3 if h==0 else 4*h+1
    def initialize(self,rng):
        if self.h==0: return np.r_[rng.normal(0,.5,2),0.]
        return np.r_[rng.normal(0,1/np.sqrt(2),2*self.h),np.zeros(self.h),rng.normal(0,1/np.sqrt(self.h),self.h),0.]
    def parts(self,t):
        h=self.h;return t[:2*h].reshape(2,h),t[2*h:3*h],t[3*h:4*h],t[-1]
    def predict(self,x,t):
        if self.h==0: return 2*expit(x@t[:2]+t[2])-1
        w,b,v,c=self.parts(t);return 2*expit(np.tanh(x@w+b)@v+c)-1
    def vg(self,x,y,t):
        f=self.predict(x,t);s=(f+1)/2;p=probability(f)
        delta=(p-y)/(p*(1-p))*(1-2e-6)*s*(1-s)/len(y)
        if self.h==0: g=np.r_[x.T@delta,delta.sum()]
        else:
            w,b,v,c=self.parts(t);z=np.tanh(x@w+b);dz=delta[:,None]*v*(1-z*z)
            g=np.r_[(x.T@dz).ravel(),dz.sum(0),z.T@delta,delta.sum()]
        return loss(f,y),g


def fit(data,target,budget,seed,checkpoints=6,rate=.025):
    model=MLP(width(target));t=model.initialize(np.random.default_rng(seed));m=np.zeros(model.d);v=m.copy()
    best=np.inf;trace=[];candidates=[];schedule=set(np.ceil(np.linspace(1,budget,checkpoints)).astype(int))
    started=time.perf_counter()
    for k in range(1,budget+1):
        check();value,g=model.vg(data.x_train,data.y_train,t)
        if value<best: best=value;best_t=t.copy()
        trace.append((k,value,best))
        if k in schedule: candidates.append(dict(nfev=k,train_loss=best,val_loss=loss(model.predict(data.x_val,best_t),data.y_val),phi=best_t.copy()))
        m=.9*m+.1*g;v=.999*v+.001*g*g;t-=rate*(m/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
    selected=min(candidates,key=lambda c:(c['val_loss'],c['nfev']))
    return model,dict(phi=selected['phi'],train_best_phi=best_t,trace=np.asarray(trace),candidates=candidates,
                      nfev=budget,budget=budget,wall_seconds=time.perf_counter()-started,best_train_loss=best,
                      selected_train_loss=selected['train_loss'],selected_val_loss=selected['val_loss'],selected_nfev=selected['nfev'],
                      target_parameters=target,actual_parameters=model.d,hidden_width=model.h,optimizer='Adam',learning_rate=rate)
