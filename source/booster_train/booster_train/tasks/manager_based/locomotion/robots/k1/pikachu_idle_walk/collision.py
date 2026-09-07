"""GPU triangle distance to the actual hollow shell, with BVH broad phase.

Distance is capped at 30 mm (outside the 10 mm penalty band). No vertex-only
sampling: edge/face intersections and edge/edge minima are included.
"""
from pathlib import Path

import numpy as np
import torch
import warp as wp


@wp.func
def point_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab, ac, ap = b-a, c-a, p-a
    d1, d2 = wp.dot(ab, ap), wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.length(p-a)
    bp = p-b
    d3, d4 = wp.dot(ab, bp), wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.length(p-b)
    vc = d1*d4-d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.length(p-(a+(d1/(d1-d3))*ab))
    cp = p-c
    d5, d6 = wp.dot(ab, cp), wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.length(p-c)
    vb = d5*d2-d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.length(p-(a+(d2/(d2-d6))*ac))
    va = d3*d6-d5*d4
    if va <= 0.0 and (d4-d3) >= 0.0 and (d5-d6) >= 0.0:
        return wp.length(p-(b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)))
    den = 1.0/(va+vb+vc)
    return wp.length(p-(a+ab*(vb*den)+ac*(vc*den)))


@wp.func
def segments(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3):
    u, v, w = q-p, b-a, p-a
    aa, bb, cc = wp.dot(u,u), wp.dot(u,v), wp.dot(v,v)
    dd, ee = wp.dot(u,w), wp.dot(v,w)
    den = aa*cc-bb*bb
    s = float(0.0)
    if den > 1.e-16:
        s = wp.clamp((bb*ee-cc*dd)/den, 0.0, 1.0)
    t = (bb*s+ee)/wp.max(cc, 1.e-16)
    if t < 0.0:
        t = 0.0
        s = wp.clamp(-dd/wp.max(aa, 1.e-16), 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = wp.clamp((bb-dd)/wp.max(aa, 1.e-16), 0.0, 1.0)
    return wp.length(w+s*u-t*v)


@wp.func
def segment_hits_triangle(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    u, v, ray = b-a, c-a, q-p
    h = wp.cross(ray, v)
    det = wp.dot(u, h)
    hit = bool(False)
    if wp.abs(det) > 1.e-12:
        inv = 1.0/det
        s = p-a
        x = wp.dot(s,h)*inv
        k = wp.cross(s,u)
        y = wp.dot(ray,k)*inv
        t = wp.dot(v,k)*inv
        hit = x >= -1.e-6 and y >= -1.e-6 and x+y <= 1.000001 and t >= 0.0 and t <= 1.0
    return hit


@wp.func
def triangle_distance(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, e: wp.vec3, f: wp.vec3):
    if (segment_hits_triangle(a,b,d,e,f) or segment_hits_triangle(b,c,d,e,f) or
        segment_hits_triangle(c,a,d,e,f) or segment_hits_triangle(d,e,a,b,c) or
        segment_hits_triangle(e,f,a,b,c) or segment_hits_triangle(f,d,a,b,c)):
        return float(0.0)
    r = wp.min(point_triangle(a,d,e,f), wp.min(point_triangle(b,d,e,f), point_triangle(c,d,e,f)))
    r = wp.min(r, wp.min(point_triangle(d,a,b,c), wp.min(point_triangle(e,a,b,c), point_triangle(f,a,b,c))))
    r = wp.min(r, wp.min(segments(a,b,d,e), wp.min(segments(a,b,e,f), segments(a,b,f,d))))
    r = wp.min(r, wp.min(segments(b,c,d,e), wp.min(segments(b,c,e,f), segments(b,c,f,d))))
    r = wp.min(r, wp.min(segments(c,a,d,e), wp.min(segments(c,a,e,f), segments(c,a,f,d))))
    return r


@wp.kernel
def clearance_kernel(mesh: wp.uint64, shell_v: wp.array(dtype=wp.vec3), shell_f: wp.array(dtype=wp.vec3i),
                     tri_a: wp.array(dtype=wp.vec3), tri_b: wp.array(dtype=wp.vec3), tri_c: wp.array(dtype=wp.vec3),
                     body_ids: wp.array(dtype=wp.int32), pos: wp.array2d(dtype=wp.vec3),
                     quat: wp.array2d(dtype=wp.quat), root_id: int, max_distance: float,
                     out: wp.array2d(dtype=float)):
    env, ti = wp.tid()
    bi = body_ids[ti]
    inv_root = wp.quat_inverse(quat[env, root_id])
    shift = pos[env, bi]-pos[env, root_id]
    a = wp.quat_rotate(inv_root, wp.quat_rotate(quat[env, bi], tri_a[ti])+shift)
    b = wp.quat_rotate(inv_root, wp.quat_rotate(quat[env, bi], tri_b[ti])+shift)
    c = wp.quat_rotate(inv_root, wp.quat_rotate(quat[env, bi], tri_c[ti])+shift)
    pad = wp.vec3(max_distance, max_distance, max_distance)
    lo = wp.vec3(wp.min(a[0],wp.min(b[0],c[0])), wp.min(a[1],wp.min(b[1],c[1])), wp.min(a[2],wp.min(b[2],c[2])))-pad
    hi = wp.vec3(wp.max(a[0],wp.max(b[0],c[0])), wp.max(a[1],wp.max(b[1],c[1])), wp.max(a[2],wp.max(b[2],c[2])))+pad
    query = wp.mesh_query_aabb(mesh, lo, hi)
    face = int(0)
    distance = max_distance
    while wp.mesh_query_aabb_next(query, face):
        ids = shell_f[face]
        d = triangle_distance(a,b,c,shell_v[ids[0]],shell_v[ids[1]],shell_v[ids[2]])
        distance = wp.min(distance,d)
    # A leg surface wholly inside the thin shell material must also be rejected.
    inside = wp.mesh_query_point(mesh, a, max_distance)
    if inside.result and inside.sign < 0.0:
        distance = 0.0
    wp.atomic_min(out, env, bi, distance)


class ShellClearance:
    def __init__(self, directory, body_names, device):
        directory = Path(directory)
        self.device = str(device)
        wp.init()
        with np.load(directory / "torso_shell_trunk.npz", allow_pickle=False) as d:
            self.v = wp.array(d["vertices"], dtype=wp.vec3, device=self.device)
            self.f = wp.array(d["faces"], dtype=wp.vec3i, device=self.device)
            self.flat_f = wp.array(d["faces"].reshape(-1), dtype=wp.int32, device=self.device)
        self.mesh = wp.Mesh(points=self.v, indices=self.flat_f, support_winding_number=True)
        with np.load(directory / "leg_collision_surfaces.npz", allow_pickle=False) as d:
            self.triangles = d["triangles"].copy()
            named_ids = [body_names.index(n) for n in d["body_names"].tolist()]
            ids = np.array(named_ids, dtype=np.int32)[d["body_indices"]]
        self.named_body_ids = named_ids
        self.triangle_body_ids = ids
        self.a = wp.array(self.triangles[:,0].copy(), dtype=wp.vec3, device=self.device)
        self.b = wp.array(self.triangles[:,1].copy(), dtype=wp.vec3, device=self.device)
        self.c = wp.array(self.triangles[:,2].copy(), dtype=wp.vec3, device=self.device)
        self.ids = wp.array(ids, dtype=wp.int32, device=self.device)
        self.root_id = body_names.index("Trunk")
        self.num_bodies = len(body_names)

    def compute(self, body_pos, body_quat, per_body=False):
        # PhysX/Isaac quaternion convention wxyz -> Warp xyzw.
        q = torch.cat((body_quat[...,1:],body_quat[...,:1]),dim=-1).contiguous()
        pos = body_pos.contiguous()
        result = torch.full((len(pos), self.num_bodies), 0.03, device=pos.device)
        wp.launch(clearance_kernel, dim=(len(pos), len(self.triangles)), inputs=[
            self.mesh.id, self.v, self.f, self.a, self.b, self.c, self.ids,
            wp.from_torch(pos,dtype=wp.vec3), wp.from_torch(q,dtype=wp.quat), self.root_id, 0.03,
            wp.from_torch(result)], device=self.device)
        # 0.1 mm conservative numerical allowance; hulls already enclose the STLs.
        return (result if per_body else result.amin(dim=1)) - 0.0001
