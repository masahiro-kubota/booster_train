"""Latch proximity at every physics step, including a bound between samples.

For linear translation and shortest-arc quaternion interpolation in the Trunk
frame, no surface point travels more than |delta p| + radius*angle. Distance to
any fixed closed set is 1-Lipschitz, so min(endpoint distances) - travel/2 is a
conservative interval bound. This is not a certification of real hardware or
an assumption of continuous collision detection by PhysX.
"""
import torch


def rotate(q, v):
    u=q[...,1:]
    uv=torch.cross(u,v,dim=-1)
    return v+2*(q[...,:1]*uv+torch.cross(u,uv,dim=-1))


def multiply(a,b):
    return torch.cat((a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdim=True),
                      a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+torch.cross(a[...,1:],b[...,1:],dim=-1)),dim=-1)


def relative_poses(p,q,root,ids):
    inv=q[:,root:root+1].expand(-1,len(ids),-1).clone();inv[...,1:]*=-1
    return rotate(inv,p[:,ids]-p[:,root:root+1]),multiply(inv,q[:,ids])


def interval_lower_bound(d0,d1,p0,p1,q0,q1,radii):
    conjugate=q0.clone();conjugate[...,1:]*=-1
    turn=multiply(conjugate,q1)
    angle=2*torch.atan2(turn[...,1:].norm(dim=-1),turn[...,0].abs())
    travel=(p1-p0).norm(dim=-1)+radii*angle
    return torch.minimum(d0,d1)-.5*travel


class SubstepClearanceGuard:
    def __init__(self,checker,num_envs):
        self.checker=checker
        ids=sorted(set(checker.named_body_ids))
        self.ids=ids
        tri_ids=checker.triangle_body_ids
        radii=[float(torch.tensor(checker.triangles[tri_ids==i]).norm(dim=-1).max()) for i in ids]
        self.radii=torch.tensor(radii,device=checker.device)
        self.minimum=torch.full((num_envs,),.0299,device=checker.device)
        self.valid=torch.zeros(num_envs,dtype=torch.bool,device=checker.device)
        self.last_distance=None
        self.last_key=None
        self.samples=0
        self.intervals=0

    def begin_control_step(self):
        self.minimum[:]=self.last_distance.amin(-1) if self.last_distance is not None else .0299

    def reset(self,env_ids):
        self.valid[env_ids]=False
        self.minimum[env_ids]=.0299
        self.last_key=None

    def observe(self,p,q,key):
        if self.last_key==key:return self.minimum
        d=self.checker.compute(p,q,per_body=True)[:,self.ids]
        rp,rq=relative_poses(p,q,self.checker.root_id,self.ids)
        if self.last_distance is not None:
            bound=interval_lower_bound(self.last_distance,d,self.last_pos,rp,self.last_quat,rq,self.radii)
            d_bound=torch.where(self.valid[:,None],bound,d)
            self.intervals+=1
        else:d_bound=d
        self.minimum=torch.minimum(self.minimum,d_bound.amin(-1))
        self.last_distance=d
        self.last_pos,self.last_quat=rp.clone(),rq.clone()
        self.valid[:]=True
        self.last_key=key
        self.samples+=1
        return self.minimum
