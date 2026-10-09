"""Independent analytic-gradient BFGS quenches; not part of optimizer ranking."""
import time
import numpy as np
from scipy.optimize import minimize
from .core import wrap,loss_and_grad
from .optimizers import BudgetExceeded
from .runtime import WorkStopped,check


def quench(model,data,budget,seed):
    rng=np.random.default_rng(seed);x0=rng.uniform(-np.pi,np.pi,model.tying.d)
    calls=0;cache={};trace=[];last=None;accepted=x0.copy();accepted_loss=None;started=time.perf_counter()
    def key(x): return np.asarray(x,dtype=float).tobytes()
    def vg(x):
        nonlocal calls,last
        check();k=key(x)
        if last is not None and last[0]==k: return last[1]
        if calls>=budget: raise BudgetExceeded()
        value,g=loss_and_grad(model,data.x_train,data.y_train,wrap(x),model.task)
        calls+=1;cache[k]=(value,g.copy());last=(k,(value,g));trace.append((calls,value,float(np.linalg.norm(g))))
        return value,g
    def callback(x):
        nonlocal accepted,accepted_loss
        accepted=np.asarray(x).copy();accepted_loss=cache[key(accepted)][0]
    message='';success=False
    try:
        accepted_loss,_=vg(x0)
        res=minimize(vg,x0,jac=True,method='BFGS',callback=callback,options=dict(maxiter=budget,gtol=1e-6))
        accepted=res.x.copy();accepted_loss=float(res.fun);success=bool(res.success);message=str(res.message)
    except BudgetExceeded: message='analytic value-gradient call budget reached'
    except WorkStopped as exc:
        exc.partial_fit=dict(nfev=calls,budget=budget,trace=np.asarray(trace),accepted_phi=wrap(accepted),accepted_loss=accepted_loss)
        raise
    value,g=cache[key(accepted)]
    return dict(phi=wrap(accepted),initial_phi=x0,accepted_loss=float(value),gradient=g,
                nfev=calls,budget=budget,trace=np.asarray(trace),success=success,message=message,
                endpoint_definition='last accepted BFGS iterate, or initial point if none accepted',
                wall_seconds=time.perf_counter()-started)
