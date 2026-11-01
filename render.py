#!/usr/bin/env python3
"""PATROL MODE — 15 s vertical concept clip built on top of the original robot-vacuum footage.

usage:
  python3 render.py SRC.mp4 TRACK.json OUT.mp4            # full render
  python3 render.py SRC.mp4 TRACK.json OUTDIR --frames 15,120,200   # debug PNGs
"""
import sys, math, json, subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, N = 1080, 1920, 30, 450
SRC, TRACK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEBUG = None
if '--frames' in sys.argv:
    DEBUG = [int(x) for x in sys.argv[sys.argv.index('--frames') + 1].split(',')]

FD = '/usr/share/fonts/truetype/jetbrains-mono/'
_fc = {}
def font(w, s):
    k = (w, s)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FD + f'JetBrainsMono-{w}.ttf', s)
    return _fc[k]

CYAN = np.array([90, 228, 255], np.float32) / 255
RED = np.array([255, 92, 98], np.float32) / 255
C8 = (90, 228, 255)
R8 = (255, 92, 98)

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lin(t, a, b): return clamp((t - a) / (b - a))
def sm(x): x = clamp(x); return x * x * (3 - 2 * x)
def eo(x): x = clamp(x); return 1 - (1 - x) ** 3
def eio(x): x = clamp(x); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x):
    x = clamp(x); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p
def col(c, a): return tuple(int(clamp(v * a, 0, 255)) for v in c)

# ---------------------------------------------------------------- tracking
tr = json.load(open(TRACK))
arr = np.array([[p[0], p[1], p[2]] if p else [np.nan] * 3 for p in tr], np.float64)
idx = np.arange(len(arr))
for c in range(3):
    m = ~np.isnan(arr[:, c]); arr[:, c] = np.interp(idx, idx[m], arr[m, c])
def gsmooth(a, s):
    r = int(3 * s); k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    return np.convolve(np.pad(a, r, mode='edge'), k, 'valid')
TR = np.stack([gsmooth(arr[:, 0], 6), gsmooth(arr[:, 1], 6), gsmooth(arr[:, 2], 10)], 1)
TRC = np.stack([gsmooth(arr[:, 0], 14), gsmooth(arr[:, 1], 14)], 1)   # crop path, calmer

def robot_at(fi):
    fi = int(clamp(fi, 0, len(TR) - 1)); return TR[fi]

# ---------------------------------------------------------------- source reading
class Reader:
    def __init__(self, start, nframes):
        self.start = start
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{start:.3f}', '-i', SRC,
                                   '-frames:v', str(nframes + 2), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                  stdout=subprocess.PIPE)
        self.last = None; self.k = 0
    def read(self):
        b = self.p.stdout.read(1280 * 720 * 3)
        if len(b) == 1280 * 720 * 3:
            self.last = np.frombuffer(b, np.uint8).reshape(720, 1280, 3)
        self.k += 1
        return self.last

def src_frame(t):
    b = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', SRC, '-frames:v', '1',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return np.frombuffer(b[:1280 * 720 * 3], np.uint8).reshape(720, 1280, 3)

def crop(img, sfi, z, fy, dx=0.0, off=(0.0, 0.0)):
    """9:16 crop following the robot. returns PIL img, robot screen pos, scale, robot radius px"""
    cx, cy = TRC[int(clamp(sfi, 0, len(TRC) - 1))]
    rx, ry, cnt = robot_at(sfi)
    cx += off[0]; cy += off[1]; rx += off[0]; ry += off[1]
    cw, ch = 405 * z, 720 * z
    x0 = clamp(cx + dx - cw / 2, 0, 1280 - cw); y0 = clamp(cy - ch * fy, 0, 720 - ch)
    im = Image.fromarray(img).transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BICUBIC)
    s = W / cw
    rad = math.sqrt(cnt * 16 / (0.6 * math.pi)) * s
    return im, ((rx - x0) * s, (ry - y0) * s), s, rad

# ---------------------------------------------------------------- grading / finishing
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = np.clip(1 - 0.85 * (((xx / W - .5) ** 2) * 1.3 + ((yy / H - .5) ** 2) * 1.0), 0, 1) ** 1.4
VIG3 = VIG[..., None]

def grade(im, dark=1.0):
    a = np.asarray(im, np.float32) / 255
    l = a @ np.array([.3, .59, .11], np.float32)
    a = a * 0.35 + l[..., None] * 0.65
    a = np.power(a, 1.25) * 0.70 * dark
    a *= np.array([0.62, 0.86, 1.14], np.float32)
    a += np.array([0.0, 0.010, 0.028], np.float32)
    return a * VIG3

def add_layer(base, ov, glow=1.0, r=7):
    """additive neon layer (PIL RGB, black = transparent) with soft bloom"""
    o = np.asarray(ov, np.float32) / 255
    small = ov.resize((W // 2, H // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(r))
    g = np.asarray(small.resize((W, H), Image.BILINEAR), np.float32) / 255
    return base + o + g * glow

def alpha_comp(base, rgba):
    a = np.asarray(rgba, np.float32) / 255
    al = a[..., 3:4]
    return base * (1 - al) + a[..., :3] * al

def grain(base, fi, amt=0.032):
    rng = np.random.default_rng(1000 + fi)
    n = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    n = np.repeat(np.repeat(n, 2, 0), 2, 1)
    return base + n[..., None] * amt * (0.35 + 0.65 * np.sqrt(np.clip(base.mean(2, keepdims=True), 0, 1)) + 0.25)

def make_label():
    f = font('Medium', 25)
    txt = 'CONCEPT · SIMULATION'
    tw = f.getlength(txt) + 6 * len(txt) * 0
    pw, ph = int(tw + 36), 50
    im = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, pw - 1, ph - 1), 6, fill=(6, 10, 16, 150), outline=(235, 242, 248, 190), width=2)
    d.text((18, ph / 2), txt, font=f, fill=(240, 246, 250, 235), anchor='lm')
    return im
LABEL = make_label()
LABEL_POS = (W - 48 - LABEL.width, 52)

# ---------------------------------------------------------------- HUD
def brackets(d, a, m=40, L=70, w=2):
    c = col(C8, 0.55 * a)
    for (x, y, sx, sy) in [(m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
        d.line([(x, y + sy * L), (x, y), (x + sx * L, y)], fill=c, width=w)

def clock(t):
    s = 7 + int(t)
    return f'02:14:{s:02d}'

def hud(d, t, a, title='PATROL MODE · 02:14', sub='UNIT 01 · NIGHT ROUTINE · LDS ONLINE', type_t0=0.25, bottom_l=None):
    brackets(d, a)
    n = int(len(title) * eo(lin(t, type_t0, type_t0 + 0.8)))
    blink = 0.35 + 0.65 * (0.5 + 0.5 * math.cos(t * 2 * math.pi * 1.1))
    if n > 0:
        d.ellipse((72, 118, 90, 136), fill=col(R8, a * blink))
        d.text((106, 127), title[:n], font=font('Regular', 36), fill=col(C8, a), anchor='lm')
        if n < len(title):
            x = 106 + font('Regular', 36).getlength(title[:n])
            d.rectangle((x + 4, 110, x + 22, 144), fill=col(C8, a * 0.8))
    sa = a * sm(lin(t, type_t0 + 0.7, type_t0 + 1.1)) * 0.6
    if sa > 0:
        d.text((106, 174), sub, font=font('Light', 22), fill=col(C8, sa), anchor='lm')
        d.line([(106, 196), (106 + 180 * sm(lin(t, type_t0 + .8, type_t0 + 1.4)), 196)], fill=col(C8, sa), width=1)
    ba = a * 0.6
    d.text((72, H - 96), bottom_l or 'REC ▪ LOW-LIGHT', font=font('Light', 22), fill=col(C8, ba), anchor='lm')
    d.text((W - 72, H - 96), clock(t), font=font('Light', 22), fill=col(C8, ba), anchor='rm')

def reticle(d, pos, rad, t, a, label='UNIT 01 · PATROLLING'):
    if a <= 0: return
    x, y = pos
    rx, ry = rad * 1.18, rad * 0.78
    rot = t * 25
    c = col(C8, 0.75 * a)
    for k in range(4):
        st = rot + k * 90 + 12
        d.arc((x - rx, y - ry, x + rx, y + ry), st, st + 60 * sm(a), fill=c, width=2)
    # leader + tag
    lx, ly = x + rx * 0.72, y - ry * 0.72
    ex, ey = lx + 70, ly - 70
    d.line([(lx, ly), (ex, ey), (ex + 40, ey)], fill=c, width=2)
    d.text((ex + 50, ey), label, font=font('Regular', 24), fill=col(C8, 0.9 * a), anchor='lm')

# ---------------------------------------------------------------- apartment plan (shared by scan + mini map)
ROOMS = {
    'HALL':    (-0.7, 0.7, -5.5, 0.9),
    'LIVING':  (0.7, 5.0, -5.5, -1.5),
    'KITCHEN': (0.7, 5.0, -1.5, 1.6),
    'BEDROOM': (-5.0, -0.7, -2.5, 1.6),
    'BATH':    (-3.0, -0.7, -5.5, -2.5),
    'STUDY':   (-5.0, -3.0, -5.5, -2.5),
}
DOORS = [(0.7, -3.5), (0.7, -0.3), (-0.7, -0.8), (-0.7, -4.0), (-4.0, -2.5), (0.0, -5.5), (3.0, -1.5)]
FURN = [(-4.6, -2.6, -1.9, 0.9, .5), (3.9, 4.8, -5.2, -2.7, .45), (2.0, 3.3, 0.0, 0.9, .75),
        (4.4, 4.95, -1.3, 1.5, .9), (1.6, 3.2, -4.9, -4.2, .4), (-2.6, -1.3, -5.3, -4.6, .5)]
HW = 1.0

def build_plan():
    segs = set()
    for (x0, x1, z0, z1) in ROOMS.values():
        for a, b in [((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)), ((x1, z1), (x0, z1)), ((x0, z1), (x0, z0))]:
            L = math.dist(a, b); n = max(1, round(L / 0.1))
            pts = [(round(a[0] + (b[0] - a[0]) * i / n, 2), round(a[1] + (b[1] - a[1]) * i / n, 2)) for i in range(n + 1)]
            for p, q in zip(pts, pts[1:]):
                mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                if any(math.dist(mid, dc) < 0.42 for dc in DOORS): continue
                segs.add(tuple(sorted([p, q])))
    segs = sorted(segs)
    deg = {}
    for p, q in segs:
        deg[p] = deg.get(p, 0) + 1; deg[q] = deg.get(q, 0) + 1
    verts = set(p for p, k in deg.items() if k == 1)
    for (x0, x1, z0, z1) in ROOMS.values():
        for p in [(x0, z0), (x1, z0), (x0, z1), (x1, z1)]:
            p = (round(p[0], 2), round(p[1], 2))
            if p in deg: verts.add(p)
    L3 = []   # (a3, b3, weight)
    for p, q in segs:
        L3.append(((p[0], 0, p[1]), (q[0], 0, q[1]), 0.55))
        L3.append(((p[0], HW, p[1]), (q[0], HW, q[1]), 1.0))
    for p in verts:
        L3.append(((p[0], 0, p[1]), (p[0], HW, p[1]), 0.8))
    for (x0, x1, z0, z1, h) in FURN:
        c = [(x, y, z) for x in (x0, x1) for y in (0, h) for z in (z0, z1)]
        for i in range(8):
            for j in range(i + 1, 8):
                if sum(1 for k in range(3) if c[i][k] != c[j][k]) == 1:
                    L3.append((c[i], c[j], 0.7))
    rng = np.random.default_rng(7)
    P = []; Wt = []
    for p, q in segs:
        for u in (0.17, 0.5, 0.83):
            x = p[0] + (q[0] - p[0]) * u; z = p[1] + (q[1] - p[1]) * u
            for y in np.linspace(0.04, HW - 0.04, 9):
                P.append((x + rng.normal(0, .012), y + rng.normal(0, .015), z + rng.normal(0, .012))); Wt.append(0.9)
    for (x0, x1, z0, z1) in ROOMS.values():
        n = int((x1 - x0) * (z1 - z0) * 26)
        for _ in range(n):
            P.append((rng.uniform(x0, x1), 0.0, rng.uniform(z0, z1))); Wt.append(0.35)
    for (x0, x1, z0, z1, h) in FURN:
        for _ in range(int((x1 - x0) * (z1 - z0) * 140)):
            P.append((rng.uniform(x0, x1), h, rng.uniform(z0, z1))); Wt.append(0.7)
    A = np.array([l[0] for l in L3], np.float32); B = np.array([l[1] for l in L3], np.float32)
    LW = np.array([l[2] for l in L3], np.float32)
    return A, B, LW, np.array(P, np.float32), np.array(Wt, np.float32), segs

LA, LB, LW, PP, PW, SEGS2D = build_plan()
ZS = 0.8   # robot position offset inside the plan during the scan
LAs = LA + np.array([0, 0, ZS], np.float32); LBs = LB + np.array([0, 0, ZS], np.float32); PPs = PP + np.array([0, 0, ZS], np.float32)
LMID_D = np.hypot((LAs[:, 0] + LBs[:, 0]) / 2, (LAs[:, 2] + LBs[:, 2]) / 2)
PP_D = np.hypot(PPs[:, 0], PPs[:, 2])

class Cam:
    def __init__(self, anchor, yaw, pitch, D, f):
        self.ax, self.ay = anchor; self.f = f
        cy_, sy_ = math.cos(yaw), math.sin(yaw); self.cy, self.sy = cy_, sy_
        cp, sp = math.cos(pitch), math.sin(pitch)
        self.C = np.array([0, D * sp, -D * cp], np.float32)
        self.F = np.array([0, -sp, cp], np.float32); self.U = np.array([0, cp, sp], np.float32)
    def __call__(self, P):
        P = np.asarray(P, np.float32)
        x = P[..., 0] * self.cy - P[..., 2] * self.sy
        z = P[..., 0] * self.sy + P[..., 2] * self.cy
        d = np.stack([x, P[..., 1], z], -1) - self.C
        xc = d[..., 0]; yc = d @ self.U; zc = d @ self.F
        return self.ax + self.f * xc / zc, self.ay - self.f * yc / zc

def draw_scan(base, t, anchor, rad, fi):
    """seg 2: apartment point cloud growing outward from the robot + rotating LiDAR beam"""
    lt = t - 3.0
    fade = sm(lin(lt, 0.0, 0.35))
    R = 0.25 + 9.5 * eio(lin(lt, 0.15, 2.6)) ** 1.15
    yaw = math.radians(lerp(-7, 4, eio(lin(lt, 0, 3))))
    cam = Cam((anchor[0], anchor[1] - rad * 0.12), yaw, math.radians(63), 9.0, 1100)
    ov = Image.new('RGB', (W, H)); d = ImageDraw.Draw(ov)
    # room floor fills, room by room
    labels = []
    for name, (x0, x1, z0, z1) in ROOMS.items():
        z0 += ZS; z1 += ZS
        far = max(math.hypot(x, z) for x in (x0, x1) for z in (z0, z1))
        near = 0 if (x0 <= 0 <= x1 and z0 <= 0 <= z1) else min(math.hypot(clamp(0, x0, x1), clamp(0, z0, z1)), 99)
        p = sm(lin(R, near + (far - near) * 0.75, far + 0.3))
        if p <= 0: continue
        sx, sy = cam(np.array([(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]))
        d.polygon(list(zip(sx.tolist(), sy.tolist())), fill=col(C8, 0.10 * p * fade))
        cx, cy = cam(np.array([((x0 + x1) / 2, 0, (z0 + z1) / 2)]))
        labels.append((name, (x1 - x0) * (z1 - z0), float(cx[0]), float(cy[0]), p))
    # lines (walls / furniture)
    vis = LMID_D < R
    ax_, ay_ = cam(LAs[vis]); bx_, by_ = cam(LBs[vis])
    br = (0.40 + 0.9 * np.exp(-(R - LMID_D[vis]) / 0.55)) * LW[vis] * fade
    for i in range(len(ax_)):
        d.line([(ax_[i], ay_[i]), (bx_[i], by_[i])], fill=col(C8, 0.85 * br[i]), width=2)
    for name, area, cx, cy, p in labels:
        d.text((cx, cy - 6), name, font=font('Regular', 24), fill=col(C8, 0.95 * p), anchor='mm')
        d.text((cx, cy + 22), f'{area:.1f} m²', font=font('Light', 18), fill=col(C8, 0.6 * p), anchor='mm')
    # scan front ring + rotating beam
    th0 = 2 * math.pi * 1.4 * lt
    ang = np.linspace(0, 2 * math.pi, 120)
    ring = np.stack([np.cos(ang) * R, np.full_like(ang, 0.02), np.sin(ang) * R], 1)
    rx_, ry_ = cam(ring)
    d.line(list(zip(rx_.tolist(), ry_.tolist())), fill=col(C8, 0.45 * fade * (1 - lin(lt, 2.3, 2.9))), width=2)
    Rb = min(R, 7.0)
    nsl = 14
    for k in range(nsl):
        a0 = th0 - (k + 1) * 0.045; a1 = th0 - k * 0.045
        pts = np.array([(0, .12, 0), (math.cos(a0) * Rb, 0.02, math.sin(a0) * Rb), (math.cos(a1) * Rb, 0.02, math.sin(a1) * Rb)])
        px, py = cam(pts)
        d.polygon(list(zip(px.tolist(), py.tolist())), fill=col(C8, 0.22 * fade * (1 - k / nsl) ** 1.6))
    ex, ey = cam(np.array([(0, .12, 0), (math.cos(th0) * Rb, .02, math.sin(th0) * Rb)]))
    d.line([(ex[0], ey[0]), (ex[1], ey[1])], fill=col(C8, 1.0 * fade), width=3)
    o = np.asarray(ov, np.float32) / 255
    # point cloud (accumulated at half res)
    pv = PP_D < R
    px, py = cam(PPs[pv])
    ix = (px / 2).astype(np.int32); iy = (py / 2).astype(np.int32)
    ok = (ix >= 0) & (ix < W // 2) & (iy >= 0) & (iy < H // 2)
    acc = np.zeros((H // 2, W // 2), np.float32)
    val = PW[pv] * (0.5 + 1.2 * np.exp(-(R - PP_D[pv]) / 0.4)) * fade
    np.add.at(acc, (iy[ok], ix[ok]), val[ok] * 0.55)
    accf = np.asarray(Image.fromarray(acc).resize((W, H), Image.BILINEAR), np.float32)
    o = o + accf[..., None] * CYAN
    # keep the robot itself readable: mask overlay where the robot is
    lum = base.mean(2)
    dd = np.hypot(xx - anchor[0], (yy - anchor[1]) / 0.7)
    rm = ((lum < 0.09) & (dd < rad * 1.25)).astype(np.float32)
    rm = np.asarray(Image.fromarray((rm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
    o *= (1 - 0.8 * rm[..., None])
    oi = Image.fromarray(np.clip(o * 255, 0, 255).astype(np.uint8))
    out = add_layer(base * (1 - 0.25 * fade), oi, glow=1.1)
    # LiDAR dome emitter on top of robot
    dome = Image.new('RGB', (W, H)); dd2 = ImageDraw.Draw(dome)
    pulse = 0.7 + 0.3 * math.sin(lt * 18)
    gx, gy = anchor[0], anchor[1] - rad * 0.12
    dd2.ellipse((gx - 10, gy - 7, gx + 10, gy + 7), fill=col(C8, pulse * fade))
    out = add_layer(out, dome, glow=2.5, r=10)
    pct = clamp(R / 9.75)
    area = sum(v[1] for v in labels if v[4] > 0.5)
    rooms = sum(1 for v in labels if v[4] > 0.5)
    return out, f'MAP {area:5.1f} m² · ROOMS {rooms}/6 · {int(pct*100):3d}%'

# ---------------------------------------------------------------- POV render (seg 3)
PH, PWd = H // 2, W // 2
FPOV = 900.0
HOR = 0.53
uu, vv = np.meshgrid(np.arange(PWd) + 0.5, np.arange(PH) + 0.5)
D0 = np.stack([(uu - PWd / 2) / (FPOV / 2), (PH * HOR - vv) / (FPOV / 2), np.ones_like(uu)], -1).astype(np.float32)
D0 /= np.linalg.norm(D0, axis=-1, keepdims=True)
XL, XR, ZB, YC = -1.8, 2.0, 6.5, 2.6
DOOR = (2.3, 3.5, 2.05)   # z0, z1, height on right wall
SOFA = ((-1.8, -0.95), (0.0, 0.8), (2.8, 4.8))
WIN = (-0.7, 0.9, 0.95, 2.1)

def pov_cam(t):
    lt = t - 6.0
    pos = np.array([0.15 + 0.03 * math.sin(lt * 1.3), 0.09 + 0.002 * math.sin(lt * 31), -0.3 + 0.16 * lt], np.float32)
    yaw = math.radians(lerp(-3, 6, eio(lin(lt, 0.6, 2.4)))) + 0.002 * math.sin(lt * 23)
    pitch = math.radians(6)
    return pos, yaw, pitch

def rot_dirs(D, yaw, pitch):
    cp, sp = math.cos(pitch), math.sin(pitch)
    y = D[..., 1] * cp + D[..., 2] * sp; z = -D[..., 1] * sp + D[..., 2] * cp
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    x = D[..., 0] * cy_ + z * sy_; z2 = -D[..., 0] * sy_ + z * cy_
    return np.stack([x, y, z2], -1)

def world_to_screen(P, pos, yaw, pitch):
    d = np.asarray(P, np.float32) - pos
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    x = d[..., 0] * cy_ - d[..., 2] * sy_; z = d[..., 0] * sy_ + d[..., 2] * cy_
    cp, sp = math.cos(pitch), math.sin(pitch)
    y = d[..., 1] * cp - z * sp; z2 = d[..., 1] * sp + z * cp
    return W / 2 + FPOV * x / z2, H * HOR - FPOV * y / z2, z2

def raycast(t, fi):
    pos, yaw, pitch = pov_cam(t)
    D = rot_dirs(D0, yaw, pitch)
    dx, dy, dz = D[..., 0], D[..., 1], D[..., 2]
    ox, oy, oz = pos
    INF = np.float32(1e9)
    def pl(num, den, cond):
        with np.errstate(divide='ignore', invalid='ignore'):
            t_ = num / den
        return np.where(cond & (t_ > 0), t_, INF)
    t_floor = pl(-oy, dy, dy < 0); t_ceil = pl(YC - oy, dy, dy > 0)
    t_l = pl(XL - ox, dx, dx < 0); t_r = pl(XR - ox, dx, dx > 0); t_b = pl(ZB - oz, dz, dz > 0)
    # door hole in right wall
    hz = oz + dz * t_r; hy = oy + dy * t_r
    hole = (hz > DOOR[0]) & (hz < DOOR[1]) & (hy < DOOR[2])
    t_r2 = np.where(hole, pl(4.6 - ox, dx, dx > 0), INF)
    t_rw = np.where(hole, INF, t_r)
    # sofa (AABB slab)
    with np.errstate(divide='ignore', invalid='ignore'):
        tx1 = (SOFA[0][0] - ox) / dx; tx2 = (SOFA[0][1] - ox) / dx
        ty1 = (SOFA[1][0] - oy) / dy; ty2 = (SOFA[1][1] - oy) / dy
        tz1 = (SOFA[2][0] - oz) / dz; tz2 = (SOFA[2][1] - oz) / dz
    tmin = np.maximum.reduce([np.minimum(tx1, tx2), np.minimum(ty1, ty2), np.minimum(tz1, tz2)])
    tmax = np.minimum.reduce([np.maximum(tx1, tx2), np.maximum(ty1, ty2), np.maximum(tz1, tz2)])
    t_s = np.where((tmax >= tmin) & (tmin > 0), tmin, INF)
    T = np.stack([t_floor, t_ceil, t_l, t_rw, t_b, t_r2, t_s])
    sid = T.argmin(0); tt = T.min(0)
    hx = ox + dx * tt; hy = oy + dy * tt; hz = oz + dz * tt
    alb = np.full(tt.shape, 0.5, np.float32)
    nd = np.ones(tt.shape, np.float32)  # lambert term
    # floor tiles
    f = sid == 0
    gx = np.abs(((hx / 0.6) % 1) - 0.5); gz = np.abs(((hz / 0.6) % 1) - 0.5)
    tile = (np.floor(hx / 0.6) * 7 + np.floor(hz / 0.6) * 13) % 5
    grout = np.clip(1 - (0.5 - np.maximum(gx, gz)) / 0.02, 0, 1) * np.exp(-tt / 5)
    alb = np.where(f, 0.50 + tile * 0.018 - grout * 0.22, alb)
    nd = np.where(f, np.clip(-dy, 0, 1) * 2.2 + 0.25, nd)
    # walls
    wl = (sid == 2) | (sid == 3) | (sid == 4)
    base_b = (hy < 0.09)
    alb = np.where(wl, np.where(base_b, 0.72, 0.62), alb)
    nd = np.where(sid == 2, np.clip(-dx, 0, 1) * 1.4 + 0.15, nd)
    nd = np.where(sid == 3, np.clip(dx, 0, 1) * 1.4 + 0.15, nd)
    nd = np.where(sid == 4, np.clip(dz, 0, 1) * 1.0, nd)
    # door jamb darkening
    jam = (sid == 3) & (((np.abs(hz - DOOR[0]) < 0.06) | (np.abs(hz - DOOR[1]) < 0.06)) & (hy < DOOR[2] + 0.06) | ((hz > DOOR[0]) & (hz < DOOR[1]) & (np.abs(hy - DOOR[2]) < 0.06)))
    alb = np.where(jam, 0.80, alb)
    alb = np.where(sid == 5, 0.25, alb); nd = np.where(sid == 5, 0.5, nd)
    alb = np.where(sid == 1, 0.45, alb); nd = np.where(sid == 1, np.clip(dy, 0, 1) * 1.2, nd)
    alb = np.where(sid == 6, 0.20, alb); nd = np.where(sid == 6, 0.9, nd)
    light = alb * nd / (1 + (tt / 2.6) ** 2) * 1.25
    # moonlit window on back wall
    win = (sid == 4) & (hx > WIN[0]) & (hx < WIN[1]) & (hy > WIN[2]) & (hy < WIN[3])
    bars = (np.abs(hx - (WIN[0] + WIN[1]) / 2) < 0.025) | (np.abs(hy - (WIN[2] + WIN[3]) / 2) < 0.025)
    moon = np.where(win & ~bars, 0.32 + 0.06 * (hy - WIN[2]), 0)
    # soft moon spill on the floor
    spill = np.where(f, 0.10 * np.exp(-((hx - 0.1) ** 2) / 0.8 - ((hz - 5.2) ** 2) / 2.2), 0)
    v = light + spill
    rgb = v[..., None] * np.array([0.50, 0.70, 0.92], np.float32) + moon[..., None] * np.array([0.55, 0.75, 1.0], np.float32)
    rng = np.random.default_rng(fi)
    rgb += rng.standard_normal(v.shape).astype(np.float32)[..., None] * 0.018
    rgb = np.clip(rgb, 0, 1) ** 0.95
    img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    depth = Image.fromarray(tt.astype(np.float32)).resize((W, H), Image.NEAREST)
    return np.asarray(img, np.float32) / 255 * VIG3 ** 0.6, np.asarray(depth, np.float32), (pos, yaw, pitch)

def person_state(t):
    t0, t1 = 6.15, 8.75
    v = 1.25
    s_end = 2.75
    lt = clamp(t - t0, 0, t1 - t0)
    s = min(v * lt, s_end) if lt < 1.6 else lerp(v * 1.6, s_end, eo((lt - 1.6) / (t1 - t0 - 1.6)))
    s = min(s, s_end)
    p0 = np.array([3.05, 0, 3.0]); p1 = np.array([0.45, 0, 2.55])
    u = s / math.dist(p0, p1)
    P = p0 + (p1 - p0) * min(u, 1)
    dir0 = (p1 - p0) / np.linalg.norm(p1 - p0)
    turn = sm(lin(t, 8.3, 9.0))
    dirv = dir0 * (1 - turn) + np.array([0.15, 0, -1]) * turn
    dirv /= np.linalg.norm(dirv)
    walk = 1 - sm(lin(t, 8.35, 8.85))
    return P, dirv, s, walk

def person_joints(t):
    P, dv, s, walk = person_state(t)
    side = np.array([dv[2], 0, -dv[0]])
    ph = 2 * math.pi * s / 1.35
    bob = 0.025 * abs(math.cos(ph)) * walk
    hip = P + np.array([0, 0.93 + bob, 0])
    neck = P + np.array([0, 1.47 + bob, 0]) + dv * 0.03
    head = P + np.array([0, 1.63 + bob, 0]) + dv * 0.04
    J = {'head': head, 'neck': neck, 'hip': hip}
    segs = []
    for sgn, phase in ((1, 0), (-1, math.pi)):
        a = 0.42 * math.sin(ph + phase) * walk
        bend = 0.55 * max(0, math.sin(ph + phase - 1.3)) * walk
        h = hip + side * 0.09 * sgn
        knee = h + dv * math.sin(a) * 0.47 + np.array([0, -math.cos(a) * 0.47, 0])
        a2 = a - bend
        ank = knee + dv * math.sin(a2) * 0.45 + np.array([0, -math.cos(a2) * 0.45, 0])
        toe = ank + dv * 0.16 + np.array([0, -0.03, 0])
        segs += [(h, knee, 0.15), (knee, ank, 0.11), (ank, toe, 0.08)]
        sh = neck + side * 0.19 * sgn + np.array([0, -0.06, 0])
        b = -0.35 * math.sin(ph + phase) * walk
        el = sh + dv * math.sin(b) * 0.29 + np.array([0, -math.cos(b) * 0.29, 0]) + side * 0.02 * sgn
        b2 = b + 0.35
        wr = el + dv * math.sin(b2) * 0.26 + np.array([0, -math.cos(b2) * 0.26, 0])
        segs += [(sh, el, 0.095), (el, wr, 0.075)]
        segs += [(h + np.array([0, 0.05, 0]), sh, 0.17)]
    segs += [(neck, hip + np.array([0, 0.05, 0]), 0.24), (neck, head, 0.1)]
    return segs, head, P

THERM = np.array([[0, 20, 0, 45], [.35, 150, 20, 95], [.6, 255, 70, 40], [.8, 255, 170, 40], [1, 255, 248, 210]], np.float32)
def thermal(v):
    v = np.clip(v, 0, 1)
    return np.stack([np.interp(v, THERM[:, 0], THERM[:, k]) for k in (1, 2, 3)], -1) / 255

def render_person(base, depth, cam, t):
    pos, yaw, pitch = cam
    segs, head, P = person_joints(t)
    mask = Image.new('L', (W, H), 0); md = ImageDraw.Draw(mask)
    pts = []
    for a, b, th in segs:
        (ax_, ay_, az), (bx_, by_, bz) = [world_to_screen(np.array(p), pos, yaw, pitch) for p in (a, b)]
        if az <= 0.1 or bz <= 0.1: continue
        wpx = FPOV * th / ((az + bz) / 2)
        md.line([(float(ax_), float(ay_)), (float(bx_), float(by_))], fill=255, width=max(1, int(wpx)))
        for (x, y) in ((ax_, ay_), (bx_, by_)):
            md.ellipse((x - wpx / 2, y - wpx / 2, x + wpx / 2, y + wpx / 2), fill=255)
        pts += [(ax_, ay_, wpx), (bx_, by_, wpx)]
    hx, hy, hz = world_to_screen(head, pos, yaw, pitch)
    hr = FPOV * 0.115 / hz
    md.ellipse((hx - hr * 0.9, hy - hr, hx + hr * 0.9, hy + hr * 1.05), fill=255)
    pd = float(np.linalg.norm(P + np.array([0, 0.9, 0]) - pos))
    m = np.asarray(mask, np.float32) / 255
    m = m * (depth > pd - 0.25)
    if m.sum() < 50:
        return base, None, pd
    ys, xs = np.nonzero(m > 0.5)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    det = sm(lin(t, 7.2, 7.5))
    # silhouette: IR-lit body, then thermal interior once detected
    mi = Image.fromarray((m * 255).astype(np.uint8))
    m_s = np.asarray(mi.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    y0, y1 = bbox[1], bbox[3]
    rel = np.clip((yy - y0) / max(1, y1 - y0), 0, 1)
    heat = 0.92 - 0.55 * rel ** 1.4
    body_ir = (0.10 + 0.10 * (1 - rel))[..., None] * np.array([0.55, 0.72, 0.9], np.float32)
    body_th = thermal(heat * 0.75) * 0.55
    body = body_ir * (1 - det) + body_th * det
    out = base * (1 - m_s[..., None]) + body * m_s[..., None]
    # thermal edge highlight
    er = np.asarray(mi.filter(ImageFilter.MinFilter(5)), np.float32) / 255
    edge = np.clip(m - er, 0, 1)
    eimg = Image.fromarray((edge * 255).astype(np.uint8))
    eg = np.asarray(eimg.filter(ImageFilter.GaussianBlur(5)), np.float32) / 255
    ecol = thermal(heat + 0.12)
    out = out + ecol * (edge * (0.25 + 0.95 * det) + eg * 1.6 * det)[..., None]
    return out, bbox, pd

def tracking_box(d, bbox, t, pd):
    if bbox is None: return
    p = lin(t, 7.2, 7.45)
    if p <= 0: return
    x0, y0, x1, y1 = bbox
    pad = 18
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = lerp(1.6, 1.0, eback(p))
    bw, bh = (x1 - x0) * k / 2, (y1 - y0) * k / 2
    x0, x1, y0, y1 = cx - bw, cx + bw, cy - bh, cy + bh
    a = sm(p * 2.5)
    c = col(R8, a)
    L = min(x1 - x0, y1 - y0) * 0.2
    for (x, y, sx, sy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        d.line([(x, y + sy * L), (x, y), (x + sx * L, y)], fill=c, width=4)
    d.rectangle((x0, y0, x1, y1), outline=col(R8, 0.25 * a), width=1)
    if p >= 0.6:
        conf = int(lerp(71, 94, eo(lin(t, 7.35, 7.7))))
        txt = f'PERSON · {conf}%'
        f = font('Medium', 30)
        tw = f.getlength(txt)
        ta = sm(lin(t, 7.33, 7.45))
        d.rectangle((x0, y0 - 52, x0 + tw + 28, y0 - 6), fill=col(R8, 0.92 * ta))
        d.text((x0 + 14, y0 - 29), txt, font=f, fill=col((10, 12, 18), 1.0 * ta / max(ta, 1e-3)) if ta > 0.5 else col(R8, 0), anchor='lm')
        d.text((x0, y1 + 30), f'ID-01 · {pd:.1f} m · THERMAL EDGE', font=font('Light', 22), fill=col(R8, 0.85 * ta), anchor='lm')

def pov_hud(d, t, a):
    brackets(d, a)
    blink = 0.35 + 0.65 * (0.5 + 0.5 * math.cos(t * 2 * math.pi * 1.1))
    d.ellipse((72, 118, 90, 136), fill=col(R8, a * blink))
    d.text((106, 127), 'ROBOT CAM · IR · FLOOR LEVEL', font=font('Regular', 32), fill=col(C8, a), anchor='lm')
    d.text((106, 174), 'h = 9 cm · LDS + ToF FUSION', font=font('Light', 22), fill=col(C8, 0.6 * a), anchor='lm')
    cx, cy = W / 2, H * HOR
    c = col(C8, 0.35 * a)
    for k in range(-6, 7):
        x = cx + k * 70
        d.line([(x, cy - (10 if k % 3 else 18)), (x, cy)], fill=c, width=1)
    d.line([(cx - 460, cy), (cx - 60, cy)], fill=c, width=1); d.line([(cx + 60, cy), (cx + 460, cy)], fill=c, width=1)
    d.text((72, H - 96), 'SCAN 10 Hz · SECTOR LIVING', font=font('Light', 22), fill=col(C8, 0.6 * a), anchor='lm')
    d.text((W - 72, H - 96), clock(t), font=font('Light', 22), fill=col(C8, 0.6 * a), anchor='rm')

# ---------------------------------------------------------------- glitch
def glitch(A, B, p, seed, strength=1.0):
    rng = np.random.default_rng(seed)
    k = 1 - abs(2 * p - 1)
    out = np.empty_like(A)
    edges = np.sort(rng.integers(0, H, 22)); edges = np.concatenate([[0], edges, [H]])
    for y0, y1 in zip(edges[:-1], edges[1:]):
        if y1 <= y0: continue
        src = B if rng.random() < p else A
        sh = int(rng.normal(0, 90 * k * strength))
        out[y0:y1] = np.roll(src[y0:y1], sh, axis=1)
    c = int(22 * k * strength)
    if c:
        out[..., 0] = np.roll(out[..., 0], c, 1); out[..., 2] = np.roll(out[..., 2], -c, 1)
    out[::4] *= 1 - 0.35 * k
    if k > 0.4:
        for _ in range(3):
            bx, by = rng.integers(0, W - 200), rng.integers(0, H - 60)
            out[by:by + rng.integers(8, 50), bx:bx + rng.integers(80, 400)] = CYAN * rng.uniform(0.3, 0.9)
    return out

# ---------------------------------------------------------------- phone (seg 4)
def rounded_mask(box, r):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).rounded_rectangle(box, r, fill=255)
    return np.asarray(m, np.float32)[..., None] / 255

PHONE = {}
def phone_assets():
    f = src_frame(10.0)
    im, pos, s, rad = crop(f, 300, 0.85, 0.42)
    wall = grade(im, 0.8)
    wall = Image.fromarray((np.clip(wall, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(14))
    wall = np.asarray(wall, np.float32) / 255 * 0.85
    fro = Image.fromarray((wall * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(40))
    fro = np.asarray(fro, np.float32) / 255 * 0.55 + np.array([0.13, 0.15, 0.19], np.float32)
    PHONE['wall'] = wall; PHONE['frost'] = fro

def plan_map(d, box, t, a, robot_p, alert_on):
    x0, y0, x1, y1 = box
    sx = (x1 - x0 - 60) / 10.0
    s = min(sx, (y1 - y0 - 60) / 7.1)
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    zc = (-5.5 + 1.6) / 2
    M = lambda x, z: (cx + x * s, cy - (z - zc) * s)
    reveal = eo(lin(t, 10.8, 11.3))
    for name, (rx0, rx1, rz0, rz1) in ROOMS.items():
        p0 = M(rx0, rz1); p1 = M(rx1, rz0)
        fillc = col(R8, 0.16 * a) if (name == 'LIVING' and alert_on > 0) else col(C8, 0.06 * a)
        d.rectangle((p0[0], p0[1], p1[0], p1[1]), fill=fillc)
    n = int(len(SEGS2D) * reveal)
    for (p, q) in SEGS2D[:n]:
        d.line([M(*p), M(*q)], fill=col(C8, 0.9 * a), width=3)
    for (fx0, fx1, fz0, fz1, h) in FURN:
        p0 = M(fx0, fz1); p1 = M(fx1, fz0)
        d.rectangle((p0[0], p0[1], p1[0], p1[1]), outline=col(C8, 0.35 * a * reveal), width=1)
    for name, (rx0, rx1, rz0, rz1) in ROOMS.items():
        if name == 'HALL': continue
        c = M((rx0 + rx1) / 2, (rz0 + rz1) / 2)
        d.text((c[0], c[1] + (34 if name == 'LIVING' else 0)), name, font=font('Regular', 20),
               fill=col(R8 if name == 'LIVING' else C8, 0.75 * a * reveal), anchor='mm')
    # robot path + robot
    path = [(0, 0.6), (0, -1.0), (0, -2.2), robot_p]
    rp = M(*robot_p)
    pts = [M(*p) for p in path if p[1] >= robot_p[1]] + [rp]
    for i in range(len(pts) - 1):
        (ax_, ay_), (bx_, by_) = pts[i], pts[i + 1]
        L = math.hypot(bx_ - ax_, by_ - ay_); nseg = int(L / 14)
        for k in range(0, nseg, 2):
            u0, u1 = k / max(nseg, 1), min(1, (k + 1) / max(nseg, 1))
            d.line([(lerp(ax_, bx_, u0), lerp(ay_, by_, u0)), (lerp(ax_, bx_, u1), lerp(ay_, by_, u1))], fill=col(C8, 0.6 * a * reveal), width=2)
    d.ellipse((rp[0] - 16, rp[1] - 16, rp[0] + 16, rp[1] + 16), fill=col((10, 14, 20), a), outline=col(C8, a * reveal), width=3)
    d.ellipse((rp[0] - 5, rp[1] - 5, rp[0] + 5, rp[1] + 5), fill=col(C8, a * reveal))
    # alert dot
    if alert_on > 0:
        ap = M(2.7, -3.3)
        ph = ((t - 11.0) * 1.4) % 1
        rr = 14 + 50 * ph
        d.ellipse((ap[0] - rr, ap[1] - rr, ap[0] + rr, ap[1] + rr), outline=col(R8, a * alert_on * (1 - ph)), width=3)
        d.ellipse((ap[0] - 14, ap[1] - 14, ap[0] + 14, ap[1] + 14), fill=col(R8, a * alert_on))
        d.text((ap[0] - 60, ap[1] - 58), 'MOTION 02:14:09', font=font('Medium', 20), fill=col(R8, a * alert_on), anchor='lm')
    return

def phone_frame(t, thumb):
    lt = t - 9.0
    base = PHONE['wall'].copy()
    ui = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
    # status + lock screen
    d.text((70, 72), '02:14', font=font('Medium', 30), fill=(240, 244, 250, 230), anchor='lm')
    d.text((W / 2, 205), 'Sunday, November 1', font=font('Regular', 34), fill=(235, 240, 248, 210), anchor='mm')
    d.text((W / 2, 340), '02:14', font=font('ExtraLight', 210), fill=(240, 246, 252, 235), anchor='mm')
    # lock glyph
    d.rounded_rectangle((W / 2 - 18, 120, W / 2 + 18, 148), 5, outline=(240, 244, 250, 200), width=3)
    d.arc((W / 2 - 12, 102, W / 2 + 12, 132), 180, 360, fill=(240, 244, 250, 200), width=3)
    # bottom quick buttons
    for x in (150, W - 150):
        d.ellipse((x - 52, H - 220, x + 52, H - 116), fill=(30, 36, 46, 150))
    d.rounded_rectangle((W / 2 - 140, H - 40, W / 2 + 140, H - 30), 5, fill=(240, 244, 250, 200))
    out = alpha_comp(base, ui)

    # card geometry: slide in -> collapsed -> expand
    slide = eback(lin(lt, 0.40, 0.85))
    ex = eio(lin(lt, 1.55, 2.05))
    cb = (40, lerp(640, 520, slide), 1040, lerp(640, 520, slide) + 250)
    eb = (40, 440, 1040, 1800)
    box = tuple(lerp(cb[i], eb[i], ex) for i in range(4))
    press = sm(lin(lt, 1.32, 1.42)) * (1 - sm(lin(lt, 1.45, 1.6)))
    sc = 1 - 0.025 * press
    bx0, by0, bx1, by1 = box
    cxm, cym = (bx0 + bx1) / 2, (by0 + by1) / 2
    box = (cxm - (bx1 - bx0) / 2 * sc, cym - (by1 - by0) / 2 * sc, cxm + (bx1 - bx0) / 2 * sc, cym + (by1 - by0) / 2 * sc)
    if lt > 0.40:
        m = rounded_mask(tuple(int(v) for v in box), 44) * sm(lin(lt, 0.40, 0.65))
        out = out * (1 - m) + (PHONE['frost'] * (1 - 0.12 * press)) * m
        # scrim behind expanded panel
        out = out * (1 - 0.35 * ex * (1 - m))
        ui = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
        bx0, by0, bx1, by1 = box
        d.rounded_rectangle(box, 44, outline=(255, 255, 255, int(40 + 30 * ex)), width=2)
        # header (shared)
        ix, iy = bx0 + 36, by0 + 34
        d.rounded_rectangle((ix, iy, ix + 64, iy + 64), 16, fill=(8, 14, 22, 255))
        d.ellipse((ix + 14, iy + 14, ix + 50, iy + 50), outline=C8 + (255,), width=3)
        d.ellipse((ix + 27, iy + 27, ix + 37, iy + 37), fill=C8 + (255,))
        d.text((ix + 86, iy + 32), 'PATROL', font=font('Medium', 26), fill=(225, 232, 240, 220), anchor='lm')
        d.text((bx1 - 36, iy + 32), 'now', font=font('Regular', 24), fill=(200, 208, 218, 170), anchor='rm')
        ca = 1 - sm(lin(lt, 1.55, 1.75))
        if ca > 0:
            A_ = int(255 * ca)
            d.text((ix, iy + 108), 'Motion detected', font=font('Medium', 36), fill=(245, 248, 252, A_), anchor='lm')
            d.text((ix, iy + 156), 'Living room · 02:14', font=font('Regular', 32), fill=(225, 232, 240, int(A_ * 0.9)), anchor='lm')
        pa = sm(lin(lt, 1.75, 2.1))
        if pa > 0:
            A_ = int(255 * pa)
            d.text((ix, iy + 118), 'Motion detected · Living room', font=font('Medium', 34), fill=(245, 248, 252, A_), anchor='lm')
            d.text((ix, iy + 166), '02:14:09 · PERSON · 94% · robot en route', font=font('Regular', 24), fill=(255, 140, 140, int(A_ * 0.95)), anchor='lm')
        fa = sm(lin(lt, 0.40, 0.65))
        if fa < 1:
            ui.putalpha(ui.getchannel('A').point(lambda v: int(v * fa)))
        out = alpha_comp(out, ui)
        # thumbnail: small (collapsed) -> big (expanded)
        if thumb is not None:
            tb_s = (int(bx1 - 36 - 172), int(by0 + 40), int(bx1 - 36), int(by0 + 40 + 172))
            tb_b = (int(bx0 + 36), int(by0 + 250), int(bx1 - 36), int(by0 + 250 + 420))
            k = eio(lin(lt, 1.6, 2.05))
            if lt < 1.6: tb_s = (tb_s[0], tb_s[1] + 52, tb_s[2], tb_s[3] + 52)
            tb = tuple(int(lerp(tb_s[i], tb_b[i], k)) for i in range(4))
            tw_, th_ = tb[2] - tb[0], tb[3] - tb[1]
            sx_, sy_, sw_ = thumb['focus']
            ar = tw_ / th_
            cw_ = sw_ * lerp(1.0, 1.6, k); ch_ = cw_ / ar
            crop_box = (sx_ - cw_ / 2, sy_ - ch_ / 2, sx_ + cw_ / 2, sy_ + ch_ / 2)
            ti = thumb['img'].transform((tw_, th_), Image.EXTENT, crop_box, Image.BICUBIC)
            ta = np.asarray(ti, np.float32) / 255
            tm = Image.new('L', (tw_, th_), 0); ImageDraw.Draw(tm).rounded_rectangle((0, 0, tw_ - 1, th_ - 1), 22, fill=255)
            tm = np.asarray(tm, np.float32)[..., None] / 255 * fa
            reg = out[tb[1]:tb[3], tb[0]:tb[2]]
            out[tb[1]:tb[3], tb[0]:tb[2]] = reg * (1 - tm) + ta * tm
        if pa > 0:
            ui = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
            mb = (bx0 + 36, by0 + 700, bx1 - 36, by0 + 1170)
            d.rounded_rectangle(mb, 22, fill=(6, 10, 16, int(200 * pa)), outline=(90, 228, 255, int(70 * pa)), width=2)
            d.text((mb[0] + 22, mb[1] + 30), 'LIVE MAP', font=font('Medium', 22), fill=(90, 228, 255, int(220 * pa)), anchor='lm')
            d.text((mb[2] - 22, mb[1] + 30), 'SIMULATED', font=font('Regular', 20), fill=(200, 208, 218, int(150 * pa)), anchor='rm')
            ov = Image.new('RGB', (W, H)); od = ImageDraw.Draw(ov)
            rpz = lerp(0.3, -2.9, eio(lin(t, 11.0, 12.0)))
            plan_map(od, (mb[0], mb[1] + 50, mb[2], mb[3]), t, pa, (0.0, rpz), sm(lin(t, 10.95, 11.15)))
            # buttons
            bt_y = by0 + 1205
            for (bxa, bxb, txt, c) in ((bx0 + 36, (bx0 + bx1) / 2 - 12, 'Live view', (90, 228, 255)), ((bx0 + bx1) / 2 + 12, bx1 - 36, 'Sound alarm', (255, 92, 98))):
                d.rounded_rectangle((bxa, bt_y, bxb, bt_y + 96), 26, outline=c + (int(220 * pa),), width=2, fill=c + (int(28 * pa),))
                d.text(((bxa + bxb) / 2, bt_y + 48), txt, font=font('Medium', 30), fill=c + (int(240 * pa),), anchor='mm')
            out = alpha_comp(out, ui)
            out = add_layer(out, ov, glow=0.6, r=5)
    # touch indicator (fingertip press + ripple)
    ta_ = sm(lin(lt, 1.15, 1.3)) * (1 - sm(lin(lt, 1.5, 1.7)))
    if ta_ > 0 or 1.45 < lt < 1.95:
        ui = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
        tx, ty = 470, 660
        r = 54 * (1 - 0.18 * press)
        if ta_ > 0:
            d.ellipse((tx - r, ty - r, tx + r, ty + r), fill=(235, 242, 250, int(95 * ta_)), outline=(255, 255, 255, int(170 * ta_)), width=2)
        rp_ = lin(lt, 1.45, 1.95)
        if 0 < rp_ < 1:
            rr = 54 + 160 * eo(rp_)
            d.ellipse((tx - rr, ty - rr, tx + rr, ty + rr), outline=(90, 228, 255, int(200 * (1 - rp_))), width=3)
        out = alpha_comp(out, ui)
    return out

# ---------------------------------------------------------------- end card (seg 5)
def end_card(out, t):
    a = sm(lin(t, 12.75, 13.35))
    if a <= 0: return out
    g = np.clip((yy - H * 0.45) / (H * 0.35), 0, 1)[..., None]
    out = out * (1 - 0.72 * a * g) * (1 - 0.15 * a)
    ui = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
    rise = 24 * (1 - eo(lin(t, 12.75, 13.5)))
    A_ = int(255 * a)
    d.text((W / 2, 1360 + rise), 'Your vacuum already', font=font('Light', 64), fill=(240, 246, 252, A_), anchor='mm')
    d.text((W / 2, 1448 + rise), 'knows your home.', font=font('Light', 64), fill=(240, 246, 252, A_), anchor='mm')
    a2 = sm(lin(t, 13.35, 13.8))
    lw = 200 * eo(lin(t, 13.2, 13.7))
    d.line([(W / 2 - lw, 1535), (W / 2 + lw, 1535)], fill=(90, 228, 255, int(180 * a)), width=2)
    d.text((W / 2, 1590), 'Not a real product feature', font=font('Regular', 30), fill=(255, 140, 145, int(230 * a2)), anchor='mm')
    return alpha_comp(out, ui)

# ---------------------------------------------------------------- main loop
def main():
    phone_assets()
    segs_src = {1: (9.0, 0), 2: (14.0, 90), 5: (26.4, 360)}
    readers = {}
    thumb = None
    last_pov = None
    last_seg2 = None
    enc = None
    if DEBUG is None:
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                                '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT],
                               stdin=subprocess.PIPE)
    else:
        os.makedirs(OUT, exist_ok=True)
    frames = range(N) if DEBUG is None else sorted(DEBUG)
    need_thumb = DEBUG is not None and any(f >= 270 for f in frames)
    if need_thumb:
        thumb = make_thumb()
    for fi in frames:
        t = fi / FPS
        if t < 3.0:
            seg = 1
        elif t < 6.0:
            seg = 2
        elif t < 9.0:
            seg = 3
        elif t < 12.0:
            seg = 4
        else:
            seg = 5
        if seg in (1, 2, 5):
            st, f0 = segs_src[seg]
            if DEBUG is not None:
                img = src_frame(st + (fi - f0) / FPS)
            else:
                if seg not in readers: readers[seg] = Reader(st, 90)
                img = readers[seg].read()
            sfi = int(round(st * FPS)) + fi - f0
            lt = (fi - f0) / FPS
            if seg == 1:
                im, pos, s, rad = crop(img, sfi, lerp(1.0, 0.94, eio(lt / 3)), 0.42, off=(-50, -25))
                base = grade(im, 1.0)
                base *= sm(lin(t, 0, 0.5))
                ov = Image.new('RGB', (W, H)); d = ImageDraw.Draw(ov)
                hud(d, t, sm(lin(t, 0.15, 0.6)))
                reticle(d, pos, rad, t, sm(lin(t, 1.2, 1.8)))
                out = add_layer(base, ov, glow=0.7)
            elif seg == 2:
                im, pos, s, rad = crop(img, sfi, lerp(0.96, 1.0, eio(lt / 3)), 0.36, off=(-38, -25))
                base = grade(im, 0.9)
                out, info = draw_scan(base, t, pos, rad, fi)
                ov = Image.new('RGB', (W, H)); d = ImageDraw.Draw(ov)
                hud(d, t, 1.0, sub='LIDAR SWEEP · BUILDING HOME MAP', type_t0=-5, bottom_l=info)
                out = add_layer(out, ov, glow=0.7)
                if t < 3.12:  # small cut glitch
                    pass
                last_seg2 = out
            else:
                im, pos, s, rad = crop(img, sfi, lerp(1.0, 0.86, eio(lt / 3)), 0.36)
                base = grade(im, 1.0)
                ov = Image.new('RGB', (W, H)); d = ImageDraw.Draw(ov)
                ha = 1 - sm(lin(t, 12.6, 13.0))
                hud(d, t, max(ha, 0.35), sub='THREAT CLEARED · RESUMING CLEAN', type_t0=-5)
                reticle(d, pos, rad, t, ha * sm(lin(t, 12.2, 12.5)), label='UNIT 01 · ON SITE')
                out = add_layer(base, ov, glow=0.7)
                out = end_card(out, t)
        elif seg == 3:
            base, depth, cam = raycast(t, fi)
            out, bbox, pd = render_person(base, depth, cam, t)
            ov = Image.new('RGB', (W, H)); d = ImageDraw.Draw(ov)
            pov_hud(d, t, 1.0)
            tracking_box(d, bbox, t, pd)
            out = add_layer(out, ov, glow=0.8)
            if DEBUG is None and abs(t - 8.4) < 1e-6 and bbox is not None:
                thumb = thumb_from(out, bbox)
            last_pov = out
        else:
            if thumb is None: thumb = make_thumb()
            out = phone_frame(t, thumb)
        # transitions (glitches)
        if 6.0 <= t < 6.2 and DEBUG is None and last_seg2 is not None:
            out = glitch(last_seg2, out, lin(t, 6.0, 6.2), fi, 0.8)
        if 9.0 <= t < 9.4:
            A = last_pov if last_pov is not None else out
            out = glitch(A, out, lin(t, 9.0, 9.4), fi, 1.2)
        if 12.0 <= t < 12.25 and DEBUG is None:
            pass
        if 11.85 <= t < 12.0:
            out = glitch(out, out * 0.4, lin(t, 11.85, 12.15), fi, 1.0)
        if 12.0 <= t < 12.2:
            out = glitch(out * 0.4, out, lin(t, 11.95, 12.2), fi, 1.0)
        if 2.93 <= t < 3.1:
            out = glitch(out, out, 0.5, fi, 0.5 * (1 - abs((t - 3.0) / 0.1)))
        out = grain(out, fi)
        img8 = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
        img8.paste(LABEL, LABEL_POS, LABEL)
        if enc:
            enc.stdin.write(img8.tobytes())
        else:
            img8.save(f'{OUT}/f{fi:03d}.png')
        if fi % 30 == 0: print('frame', fi, flush=True)
    if enc:
        enc.stdin.close(); enc.wait()

def thumb_from(out, bbox):
    img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    x0, y0, x1, y1 = bbox
    return {'img': img, 'focus': ((x0 + x1) / 2, (y0 + y1) / 2, (y1 - y0) * 1.25)}

def make_thumb():
    t = 8.4
    base, depth, cam = raycast(t, 252)
    out, bbox, pd = render_person(base, depth, cam, t)
    ov = Image.new('RGB', (W, H)); d = ImageDraw.Draw(ov)
    tracking_box(d, bbox, t, pd)
    out = add_layer(out, ov, glow=0.8)
    return thumb_from(out, bbox)

if __name__ == '__main__':
    main()
