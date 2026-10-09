"""Post-hoc landscape and data expressivity. This module never optimizes.

Methods adapted from the supplied landscape script, with QML losses, exact
shared-angle derivatives, multiplicity normalization and projected large Hessians.
"""
import numpy as np
from scipy.stats import spearmanr
from .core import loss,loss_and_grad,wrap
from .models import Model,collective_rotate,pauli
from .datasets import graph_relabel
from .runtime import check


def entropy(counts):
    p=np.asarray(counts,float);p=p[p>0];p/=p.sum()
    h=float(-np.sum(p*np.log(p)))
    return dict(entropy=h,normalized_entropy=h/np.log(len(p)) if len(p)>1 else 0.,effective_clusters=float(np.exp(h)))


def clusters(points,values,parameter_tol=.5,loss_atol=1e-4,loss_rtol=1e-3,weights=None,periods=None):
    groups=[];assignment=np.full(len(points),-1,int)
    for i in np.argsort(values):
        found=None
        for j,members in enumerate(groups):
            k=members[0];tol=loss_atol+loss_rtol*max(abs(values[i]),abs(values[k]))
            if abs(values[i]-values[k])>tol: continue
            distances=[]
            for k in members[:8]:
                delta=wrap(points[i]-points[k]) if periods is None else (points[i]-points[k]+periods/2)%periods-periods/2
                distances.append(np.sqrt(np.mean(delta*delta)) if weights is None else np.sqrt(np.sum(weights*delta*delta)/np.sum(weights)))
            if min(distances)<=parameter_tol: found=j;break
        if found is None: found=len(groups);groups.append([])
        groups[found].append(int(i));assignment[i]=found
    counts=[len(g) for g in groups]
    return dict(n_clusters=len(groups),assignment=assignment.tolist(),counts=counts,
                best_observed_cluster_occupancy=float(counts[assignment[int(np.argmin(values))]]/len(points)),
                **entropy(counts))


def spectrum(h,relative=1e-5,absolute=1e-7):
    vals=np.linalg.eigvalsh(h);tol=max(absolute,relative*max(float(np.max(abs(vals))),absolute))
    positive=vals[vals>tol];negative=vals[vals < -tol];diagonal=np.diag(np.diag(h));off=h-diagonal
    coupling_threshold=max(1e-8,1e-3*float(np.max(abs(h))))
    edges=int(np.sum(np.triu(abs(h)>coupling_threshold,1)));possible=len(h)*(len(h)-1)//2
    return dict(eigen_min=float(vals.min()),eigen_max=float(vals.max()),eigen_median=float(np.median(vals)),
                n_positive=len(positive),n_negative=len(negative),n_near_zero=int(np.sum(abs(vals)<=tol)),eigen_threshold=tol,
                positive_condition=float(positive.max()/positive.min()) if len(positive) else None,
                offdiagonal_frobenius_ratio=float(np.linalg.norm(off)/max(np.linalg.norm(h),1e-15)),
                coupling_threshold=coupling_threshold,coupling_edges=edges,coupling_density=edges/possible if possible else 0.),vals


def hessian(model,x,y,phi,rng,full_limit=30,projection=10,h=.002):
    d=model.tying.d;inverse_root=1/np.sqrt(model.tying.multiplicity)
    q=np.eye(d) if d<=full_limit else np.linalg.qr(rng.normal(size=(d,min(d,projection))))[0]
    estimates=[];evaluations=0
    for step in (h,h/2):
        columns=[]
        for direction in q.T:
            check();_,gp=loss_and_grad(model,x,y,phi+step*inverse_root*direction,model.task)
            _,gm=loss_and_grad(model,x,y,phi-step*inverse_root*direction,model.task)
            columns.append(q.T@(inverse_root*(gp-gm))/(2*step));evaluations+=2
        a=np.array(columns).T;estimates.append((a+a.T)/2)
    residual=float(np.linalg.norm(estimates[0]-estimates[1])/max(np.linalg.norm(estimates[1]),1e-8))
    stats,eigenvalues=spectrum(estimates[1])
    stats.update(mode='full' if d<=full_limit else 'projected',ambient_dimension=d,diagnosed_dimension=q.shape[1],
                 fd_step_physical=h/2,fd_relative_residual=residual,fd_reliable=residual<=.05,
                 jacobian_evaluations=evaluations,
                 coupling_interpretation='reduced physical-metric coordinates' if d<=full_limit else 'random projected basis; not parameter separability')
    return stats,dict(hessian=estimates[1],hessian_coarse=estimates[0],hessian_basis=q,eigenvalues=eigenvalues)


def landscape(model,data,endpoints,incumbents,cfg,seed):
    """Read quenches/training incumbents; evaluate only, with raw probes saved."""
    if not endpoints: raise ValueError('Independent quench endpoints are required.')
    rng=np.random.default_rng(seed);x,y=data.x_train,data.y_train;d=model.tying.d;m=model.tying.multiplicity
    root=np.sqrt(m);inverse_root=1/root;evals=0;jacs=0;arrays={}
    ep=np.array([e['phi'] for e in endpoints]);values=np.array([e['accepted_loss'] for e in endpoints])
    arrays.update(endpoint_phi=ep,endpoint_loss=values,endpoint_gradient=np.array([e['gradient'] for e in endpoints]))
    anchor=ep[np.argmin(values)].copy();source='best independent quench endpoint'
    if incumbents:
        inc=np.array([e['phi'] for e in incumbents]);iv=np.array([e['loss'] for e in incumbents]);arrays.update(incumbent_phi=inc,incumbent_loss=iv)
        if iv.min()<values.min(): anchor=inc[np.argmin(iv)].copy();source='best training incumbent across primary optimizers'
    anchor_loss=loss(model.predict(x,anchor),y,model.task);evals+=1;arrays['anchor_phi']=anchor
    cluster=clusters(ep,values);sensitivity=[]
    known_periods=np.full(d,2*np.pi)
    if model.spec.family=='su2':
        for group in range(d):
            local_counts=[sum(model.tying.ids[o[3]]==group for o in model.ops[b:b+3]) for b in range(0,len(model.ops),3)]
            if all(c in (0,3) for c in local_counts):known_periods[group]=np.pi
    arrays['known_parameter_periods']=known_periods
    levels=[];loss_assignment=np.zeros(len(values),int)
    for i in np.argsort(values):
        if not levels or abs(values[i]-values[levels[-1][0]])>1e-4+1e-3*max(abs(values[i]),abs(values[levels[-1][0]])):
            levels.append([])
        levels[-1].append(int(i));loss_assignment[i]=len(levels)-1
    distinct_clusters=[len(set(cluster['assignment'][i] for i in level)) for level in levels]
    arrays['endpoint_loss_level']=loss_assignment
    for tol in (.25,.5,1.):
        sensitivity.append(dict(parameter_tol=tol,**clusters(ep,values,parameter_tol=tol)))
    grad_raw=[];grad_norm=[];probe_phi=[];probe_losses=[]
    for _ in range(cfg.get('gradient_samples',16)):
        check();phi=rng.uniform(-np.pi,np.pi,d);v,g=loss_and_grad(model,x,y,phi,model.task);jacs+=1
        probe_phi.append(phi);probe_losses.append(v);grad_raw.append(g);grad_norm.append(np.linalg.norm(g*inverse_root)/np.sqrt(d))
    arrays.update(gradient_probe_phi=np.asarray(probe_phi),gradient_probe_raw=np.asarray(grad_raw),gradient_probe_loss=np.asarray(probe_losses))
    global_phi=[];global_values=[];distances=[];raw_distances=[]
    for _ in range(cfg.get('global_samples',32)):
        check();phi=rng.uniform(-np.pi,np.pi,d);v=loss(model.predict(x,phi),y,model.task);evals+=1;delta=wrap(phi-anchor)
        global_phi.append(phi);global_values.append(v);distances.append(np.sqrt(np.sum(m*delta*delta)/m.sum()));raw_distances.append(np.sqrt(np.mean(delta*delta)))
    rho=spearmanr(distances,global_values).statistic if np.ptp(global_values)>1e-12 else np.nan
    arrays.update(global_phi=np.asarray(global_phi),global_loss=np.asarray(global_values),global_physical_distance=np.asarray(distances),global_torus_rms=np.asarray(raw_distances))
    anchors=[anchor,rng.uniform(-np.pi,np.pi,d)];hessians=[]
    release=[];independent=Model(model.spec,'untied');a=model.tying.matrix()
    for i,point in enumerate(anchors):
        check();value,g=loss_and_grad(model,x,y,point,model.task);jacs+=1
        hs,ha=hessian(model,x,y,point,rng,cfg.get('full_hessian_limit',30),cfg.get('hessian_projection',10));jacs+=hs['jacobian_evaluations']
        hs.update(point='best_observed' if i==0 else 'deterministic_random',loss=value,
                  physical_gradient_rms=float(np.linalg.norm(g*inverse_root)/np.sqrt(d)),
                  stationary_minimum_proxy=bool(np.linalg.norm(g*inverse_root)/np.sqrt(d)<1e-4 and hs['mode']=='full' and hs['fd_reliable'] and hs['n_negative']==0))
        hessians.append(hs)
        arrays.update({f'{k}_{i}':v for k,v in ha.items()});arrays[f'hessian_phi_{i}']=point;arrays[f'hessian_gradient_{i}']=g
        # At a restricted stationary point A^T g_theta can vanish while a
        # gradient normal to the tying subspace remains. This distinguishes
        # constraint-induced traps from stationarity of the untied circuit.
        theta=model.tying.expand(point);_,physical_g=loss_and_grad(independent,x,y,theta,model.task);jacs+=1
        tangent=a@((a.T@physical_g)/m);normal=physical_g-tangent
        absolute_sums=a.T@abs(physical_g);group_cancellation=1-abs(a.T@physical_g)/np.maximum(absolute_sums,1e-15)
        active_cancellation=absolute_sums>1e-8
        group_cancellation[~active_cancellation]=0.
        release.append(dict(point=hs['point'],untied_gradient_rms=float(np.linalg.norm(physical_g)/np.sqrt(model.p)),
                            normal_gradient_rms=float(np.linalg.norm(normal)/np.sqrt(model.p)),
                            normal_gradient_fraction=float(np.linalg.norm(normal)/max(np.linalg.norm(physical_g),1e-15)),
                            shared_group_cancellation_mean=float(np.mean(group_cancellation[active_cancellation])) if np.any(active_cancellation) else None,
                            cancellation_active_groups=int(np.sum(active_cancellation)),cancellation_absolute_gradient_threshold=1e-8,
                            chain_rule_residual=float(np.linalg.norm(a.T@physical_g-g))))
        hs['normal_gradient_rms']=release[-1]['normal_gradient_rms']
        hs['restricted_trap_candidate']=bool(hs['stationary_minimum_proxy'] and hs['normal_gradient_rms']>1e-4)
        arrays[f'untied_gradient_{i}']=physical_g;arrays[f'normal_gradient_{i}']=normal;arrays[f'group_cancellation_{i}']=group_cancellation;arrays[f'cancellation_group_active_{i}']=active_cancellation
    # The parameter clusters have gauge redundancies. Save an independent
    # functional-distance check on the same training examples.
    endpoint_predictions=np.array([model.predict(x,p) for p in ep]);evals+=len(ep)
    arrays['endpoint_predictions']=endpoint_predictions
    functional=[]
    for i in np.argsort(values):
        found=False
        for group in functional:
            if np.sqrt(np.mean((endpoint_predictions[i]-endpoint_predictions[group[0]])**2))<=.02:
                group.append(int(i));found=True;break
        if not found: functional.append([int(i)])
    grid=np.linspace(-np.pi,np.pi,cfg.get('line_points',65));lines=[]
    for j in range(cfg.get('lines',2)):
        direction=rng.normal(size=d);direction/=np.linalg.norm(direction);curve=[];points=[]
        for t in grid:
            check();point=anchor+t*inverse_root*direction;points.append(wrap(point));curve.append(loss(model.predict(x,point),y,model.task));evals+=1
        a=np.asarray(curve);turns=int(np.sum((a[1:-1]<a[:-2])&(a[1:-1]<a[2:])))
        lines.append(dict(line=j,sampled_local_minima=turns,range=float(np.ptp(a)),roughness=float(np.mean(abs(np.diff(a,2)))),physical_step=float(grid[1]-grid[0])))
        arrays[f'line_phi_{j}']=np.array(points);arrays[f'line_loss_{j}']=a;arrays[f'line_direction_{j}']=direction
    arrays['line_grid']=grid
    grad_norm=np.asarray(grad_norm);ep_grad=np.linalg.norm(arrays['endpoint_gradient']/root,axis=1)/np.sqrt(d)
    arrays['endpoint_physical_gradient_rms']=ep_grad
    record=dict(anchor_source=source,anchor_loss=anchor_loss,endpoint_cluster=cluster,
                endpoint_cluster_known_periods=clusters(ep,values,periods=known_periods),known_parameter_periods=known_periods.tolist(),
                endpoint_cluster_physical_metric=clusters(ep,values,weights=m),cluster_tolerance_sensitivity=sensitivity,
                functional_endpoint_clusters=len(functional),functional_cluster_counts=[len(g) for g in functional],
                empirical_loss_levels=len(levels),same_loss_distinct_parameter_cluster_levels=int(np.sum(np.asarray(distinct_clusters)>1)),
                parameter_clusters_per_loss_level=distinct_clusters,
                quench_stationary_fraction=float(np.mean(ep_grad<1e-4)),quench_budget_limited_fraction=float(np.mean([e['nfev']>=e['budget'] for e in endpoints])),
                best_observed_endpoint_gap_median=float(np.median(values-anchor_loss)),
                gradient_physical_rms_median=float(np.median(grad_norm)),gradient_physical_rms_q10=float(np.quantile(grad_norm,.1)),
                low_gradient_fraction_1e4=float(np.mean(grad_norm<1e-4)),loss_distance_spearman=float(rho) if np.isfinite(rho) else None,
                hessians=hessians,release_gradient_diagnostics=release,
                best_normal_gradient_rms=release[0]['normal_gradient_rms'],best_shared_group_cancellation=release[0]['shared_group_cancellation_mean'],
                lines=lines,diagnostic_forward_calls=evals,diagnostic_jacobian_calls=jacs,diagnostic_untied_jacobian_calls=2,
                interpretation='finite-start endpoint occupancy, not basin volume or a global-optimum certificate; Hessian couplings depend on coordinates')
    return record,arrays


def symmetry(model,x,phi,rng):
    family=model.spec.family;n=model.spec.n;x=x[:min(12,len(x))];base=model.predict(x,phi);errors=[]
    if family=='su2':
        for _ in range(4):
            angle=rng.uniform(-np.pi,np.pi,3);errors.append(np.max(abs(base-model.predict(collective_rotate(x,n,angle),phi))))
        exact=model.tying.name in ('su2','su2_spatial')
        # Gate-level Lie-generator audit. Coincident XX, YY, ZZ rotations
        # commute, so their tied generator is the isotropic exchange sum.
        eye=np.eye(1<<n,dtype=complex);js=[sum(pauli(eye,n,a,(q,)).T/2 for q in range(n)) for a in 'xyz']
        norms=[]
        for b in range(0,len(model.ops),3):
            ops=model.ops[b:b+3];ids=[model.tying.ids[o[3]] for o in ops]
            for group in set(ids):
                g=sum(pauli(eye,n,o[1],o[2]).T/2 for o,i in zip(ops,ids) if i==group)
                for j in js: norms.append(float(np.linalg.norm(g@j-j@g)/max(np.linalg.norm(g)*np.linalg.norm(j),1e-15)))
        return dict(group='collective SU(2)',exact_architecture_certificate=exact,prediction_error_max=float(max(errors)),
                    local_generator_commutator_max=float(max(norms)),readout='singlet projector; invariant under collective rotations')
    if family=='graph':
        for _ in range(4): errors.append(np.max(abs(base-model.predict(graph_relabel(x,n,rng.permutation(n)),phi))))
        return dict(group='joint graph/qubit relabeling S_n',exact_architecture_certificate=model.tying.name in ('node_edge','layer_shared'),prediction_error_max=float(max(errors)),
                    readout='mean Z; permutation-invariant',split_orbits_kept_together=True)
    if family=='reupload':
        error=float(np.max(abs(base-model.predict(x[:,::-1],phi))))
        return dict(group='two-feature swap via cyclic qubit shift',exact_architecture_certificate=n%2==0 and model.tying.name in ('space','global_axis'),prediction_error_max=error)
    return dict(group='none asserted',exact_architecture_certificate=False,
                motivation={'qcnn':'homogeneous local kernels; pooling can break one-site translation equivariance',
                            'autoencoder':'inverse decoding and repeated encoder refinement; no spatial symmetry asserted',
                            'recurrent':'stationary temporal update; not time-permutation invariance'}[family])


def expressivity(model,data,cfg,seed):
    rng=np.random.default_rng(seed);d=model.tying.d;root=np.sqrt(model.tying.multiplicity);x=data.x_train[:min(48,len(data.x_train))]
    outputs=[];phis=[];jacobians=[];ranks=[];svs=[];syms=[]
    for i in range(cfg.get('parameter_samples',12)):
        check();phi=rng.uniform(-np.pi,np.pi,d);phis.append(phi)
        if i<cfg.get('jacobian_samples',4):
            f,j=model.predict(x,phi,jac=True);j=j/root[None];singular=np.linalg.svd(j,compute_uv=False)
            threshold=max(1e-8,1e-6*(singular[0] if len(singular) else 0.));ranks.append(int(np.sum(singular>threshold)));svs.append(singular);jacobians.append(j)
        else: f=model.predict(x,phi)
        outputs.append(f)
        if i<2: syms.append(symmetry(model,x,phi,rng))
    outputs=np.asarray(outputs);centered=outputs-outputs.mean(0);gram=centered@centered.T/len(x);s=np.linalg.eigvalsh(gram)
    active=s>max(1e-10,1e-6*max(float(s.max()),0.));effective=float(s.sum()**2/np.sum(s*s)) if np.sum(s*s)>1e-24 else 0.
    arrays=dict(probe_x=x,parameter_phi=np.asarray(phis),outputs=outputs,output_gram=gram,output_gram_eigenvalues=s,
                physical_metric_jacobians=np.asarray(jacobians),jacobian_singular_values=np.asarray(svs))
    record=dict(sampled_function_rank=int(active.sum()),sampled_function_effective_rank=effective,
                jacobian_ranks=ranks,jacobian_rank_max=max(ranks) if ranks else None,free_parameters=d,physical_parameters=model.p,
                output_variance_mean=float(np.mean(np.var(outputs,axis=0))),symmetry_audits=syms,
                interpretation='sampled local/function ranks on training inputs; not a global expressivity or Lie-controllability certificate')
    if model.spec.family=='reupload':
        spec=model.reference.spec;bound=spec.bandwidth;points=2*max(bound)+3
        grid=np.arange(points)*2*np.pi/points-np.pi;xx=np.stack(np.meshgrid(grid,grid,indexing='ij'),axis=-1).reshape(-1,2)
        coefficients=[]
        for phi in phis:
            check();f=model.predict(xx,phi).reshape(points,points);coefficients.append(np.fft.fftshift(np.fft.fft2(f))/points**2)
        coefficients=np.asarray(coefficients);freq=np.fft.fftshift(np.fft.fftfreq(points,d=1/points)).astype(int)
        support=np.max(abs(coefficients),axis=0)>1e-7;outside=(abs(freq[:,None])>bound[0])|(abs(freq[None,:])>bound[1])
        arrays.update(fourier_coefficients=coefficients,fourier_frequencies=freq,fourier_grid=xx)
        record.update(encoding_frequency_bound=list(bound),sampled_fourier_support=int(support.sum()),
                      maximum_out_of_bound_coefficient=float(np.max(abs(coefficients[:,outside]))) if np.any(outside) else 0.,fourier_threshold=1e-7,
                      fourier_note='sharing changes accessible coefficients; the fixed encoding sets the frequency bound')
    return record,arrays
