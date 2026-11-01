#!/usr/bin/env python3
"""PATROL MODE v2 — 16:9. Left: the original footage (no grade) with a person (dark legs silhouette)
standing still at the doorway. Right: an iPhone screen recording of the Patrol app (HUD, map, detection).

usage:
  python3 render_v2.py SRC.mp4 DATA_DIR OUT.mp4
  python3 render_v2.py SRC.mp4 DATA_DIR OUTDIR --frames 30,90,150
DATA_DIR holds track.json (robot), poses.json (camera pose per frame, from tools/floor_pose.py)
"""
import sys, math, json, subprocess, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, N = 1920, 1080, 30, 450
SRC, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEBUG = [int(x) for x in sys.argv[sys.argv.index('--frames') + 1].split(',')] if '--frames' in sys.argv else None

FD = '/usr/share/fonts/truetype/jetbrains-mono/'
_fc = {}
def font(w, s):
    if (w, s) not in _fc: _fc[(w, s)] = ImageFont.truetype(FD + f'JetBrainsMono-{w}.ttf', int(s))
    return _fc[(w, s)]
C8 = (90, 228, 255); R8 = (255, 92, 98); WH = (238, 243, 248)

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lin(t, a, b): return clamp((t - a) / (b - a))
def sm(x): x = clamp(x); return x * x * (3 - 2 * x)
def eo(x): x = clamp(x); return 1 - (1 - x) ** 3
def eio(x): x = clamp(x); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x):
    x = clamp(x); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p
def rgba(c, a): return (int(c[0]), int(c[1]), int(c[2]), int(clamp(a) * 255))

# ------------------------------------------------------------------ data
tr = json.load(open(f'{DATA}/track.json'))
arr = np.array([[p[0], p[1]] if p else [np.nan] * 2 for p in tr], np.float64)
for c in range(2):
    m = ~np.isnan(arr[:, c]); i_ = np.arange(len(arr)); arr[:, c] = np.interp(i_, i_[m], arr[m, c])
def gsm(a, s):
    r = int(3 * s); k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    return np.convolve(np.pad(a, r, mode='edge'), k, 'valid')
ROB = np.stack([gsm(arr[:, 0], 5) - 40, gsm(arr[:, 1], 5) - 22], 1)   # tracker sits low-right of centre
POSES = {int(k): v for k, v in json.load(open(f'{DATA}/poses.json')).items()}
FOC = 1900.0
K = np.array([[FOC, 0, 640], [0, FOC, 360], [0, 0, 1]], np.float64)

def pose(fi):
    fi = min(fi, max(POSES)); a = int(math.floor(fi)); b = min(a + 1, max(POSES)); u = fi - a
    rv = np.array(POSES[a][0]) * (1 - u) + np.array(POSES[b][0]) * u
    tv = np.array(POSES[a][1]) * (1 - u) + np.array(POSES[b][1]) * u
    R, _ = cv2.Rodrigues(rv); return R, tv

# ------------------------------------------------------------------ person (legs) — world: x left, y up, z forward
FEET = np.array([0.10, 0.0, 0.74])
SECT = [(0.06, 0.046), (0.09, 0.050), (0.16, 0.055), (0.29, 0.066), (0.41, 0.063), (0.50, 0.060), (0.62, 0.073), (0.74, 0.083), (0.86, 0.088)]

def leg_axis(side, y):
    # side = +1 left leg (world +x), -1 right leg
    x = FEET[0] + side * lerp(0.10, 0.085, y / 0.86)
    return np.array([x, y, FEET[2] - 0.02 + 0.01 * y])

def foot_pts(side):
    yaw = side * 0.18
    out = []
    for k, (u, r) in enumerate([(-0.07, 0.042), (-0.03, 0.047), (0.03, 0.046), (0.09, 0.042), (0.15, 0.035), (0.19, 0.028)]):
        x = FEET[0] + side * 0.10 + math.sin(yaw) * u
        out.append((np.array([x, 0.038 if k < 2 else 0.03, FEET[2] + math.cos(yaw) * u]), r))
    return out

def proj(R, t, P):
    Xc = R @ P + t
    return np.array([FOC * Xc[0] / Xc[2] + 640, FOC * Xc[1] / Xc[2] + 360]), Xc[2]

def draw_chain(d, R, t, chain):
    """chain: list of (P3, radius) -> filled capsules"""
    pp = []
    for P, r in chain:
        q, z = proj(R, t, P)
        if z < 0.06: pp.append(None); continue
        pp.append((q, FOC * r / z))
    for a, b in zip(pp, pp[1:]):
        if a is None or b is None: continue
        (pa, ra), (pb, rb) = a, b
        dv = pb - pa; L = np.linalg.norm(dv) + 1e-6; n = np.array([-dv[1], dv[0]]) / L
        poly = [tuple(pa + n * ra), tuple(pb + n * rb), tuple(pb - n * rb), tuple(pa - n * ra)]
        d.polygon(poly, fill=255)
    for p in pp:
        if p is None: continue
        q, r = p
        d.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=255)

_rn = np.random.default_rng(3).standard_normal((720, 1280)).astype(np.float32)
FABRIC = cv2.GaussianBlur(_rn, (0, 0), sigmaX=9, sigmaY=2.2) * 14 + cv2.GaussianBlur(_rn, (0, 0), 1.2) * 3

def person_layer(fi, prev_feet):
    """returns (rgb float, alpha float, shadow-multiplier float, bbox, feet_px)"""
    R, t = pose(fi)
    S_img = Image.new('L', (1280, 720), 0); dS = ImageDraw.Draw(S_img)
    for side in (1, -1):
        draw_chain(dS, R, t, foot_pts(side))
    S = np.asarray(S_img, np.float32) / 255
    M = np.zeros((720, 1280), np.float32)
    col = np.zeros((720, 1280, 3), np.float32)
    albedo = np.array([0.20, 0.205, 0.225], np.float32)
    for side in (1, -1):
        mL = Image.new('L', (1280, 720), 0)
        draw_chain(ImageDraw.Draw(mL), R, t, [(leg_axis(side, y), r) for y, r in SECT])
        m = np.asarray(mL, np.float32) / 255
        ys, xs = np.nonzero(m > 0)
        if len(xs) == 0: continue
        pts = []
        for y, r in SECT:
            q, z = proj(R, t, leg_axis(side, y))
            if z > 0.06: pts.append((q, FOC * r / z, y))
        P = np.stack([xs, ys], 1).astype(np.float32)
        best = np.full(len(xs), 1e9, np.float32); U = np.zeros(len(xs), np.float32); V = np.zeros(len(xs), np.float32)
        for (qa, ra, ya), (qb, rb, yb) in zip(pts, pts[1:]):
            dv = (qb - qa).astype(np.float32); L2 = float(dv @ dv) + 1e-6
            sp = np.clip(((P - qa) @ dv) / L2, 0, 1)
            off = P - (qa + sp[:, None] * dv)
            n = np.array([-dv[1], dv[0]], np.float32) / math.sqrt(L2)
            sd = off @ n; rr = ra + (rb - ra) * sp
            dd = np.abs(sd) / rr
            upd = dd < best
            best = np.where(upd, dd, best); U = np.where(upd, sd / rr, U); V = np.where(upd, ya + (yb - ya) * sp, V)
        U = np.clip(U, -1, 1)
        nz = np.sqrt(np.clip(1 - U ** 2, 0, 1))
        diff = 0.16 + 1.0 * np.clip(nz * 0.8 - U * 0.4, 0, 1) ** 1.3
        ao = 0.55 + 0.45 * np.clip((V - 0.04) / 0.22, 0, 1)
        fold = 1 - 0.22 * np.clip(1 - (V - 0.07) / 0.14, 0, 1) * (0.5 + 0.5 * np.sin(V * 170 + U * 2.5))
        fold *= 1 - 0.14 * np.exp(-((V - 0.47) / 0.05) ** 2) * (0.5 + 0.5 * np.sin(U * 5 + V * 80))
        fold *= 1 + 0.06 * np.sin(V * 23 + U * 1.3)
        rim = np.clip(U, 0, 1) ** 5 * 0.30 + np.clip(-U, 0, 1) ** 8 * 0.05
        c = albedo[None] * (diff * ao * fold)[:, None] + rim[:, None] * np.array([0.85, 0.86, 0.84], np.float32)
        c = c + (FABRIC[ys, xs] / 255 * 0.8)[:, None]
        col[ys, xs] = c
        M = np.maximum(M, m)
    if M.sum() + S.sum() < 30:
        return None
    sockc = np.array([0.06, 0.06, 0.065], np.float32)
    col = col * M[..., None] + sockc * (1 - M[..., None])
    A = np.clip(M + S, 0, 1)
    # shadows on the floor: contact + soft cast toward camera
    sh = Image.new('L', (1280, 720), 0); dsh = ImageDraw.Draw(sh)
    cast = Image.new('L', (1280, 720), 0); dc = ImageDraw.Draw(cast)
    for side in (1, -1):
        c0 = FEET + np.array([side * 0.115, 0, 0.05])
        ring = [c0 + np.array([math.cos(a) * 0.13, 0, math.sin(a) * 0.17]) for a in np.linspace(0, 2 * math.pi, 24)]
        pr = [proj(R, t, p) for p in ring]
        if all(z > 0.06 for _, z in pr): dsh.polygon([tuple(q) for q, _ in pr], fill=255)
        chain = []
        for y, r in SECT:
            P = leg_axis(side, y); chain.append((np.array([P[0] + 0.25 * y, 0, P[2] - 0.55 * y]), r * 1.1))
        draw_chain(dc, R, t, chain)
    shm = cv2.GaussianBlur(np.asarray(sh, np.float32) / 255, (0, 0), 16)
    cam = cv2.GaussianBlur(np.asarray(cast, np.float32) / 255, (0, 0), 30)
    mult = 1 - 0.55 * shm - 0.28 * cam
    # motion blur matched to camera motion
    fp, z = proj(R, t, FEET + np.array([0, 0.3, 0]))
    if prev_feet is not None:
        dv = fp - prev_feet; L = float(np.linalg.norm(dv))
        if L > 2:
            k = int(min(L * 0.6, 40)) | 1
            ker = np.zeros((k, k), np.float32)
            cx = k // 2; ux, uy = dv / L
            for s in np.linspace(-cx, cx, k * 2):
                ker[int(round(cx + uy * s)), int(round(cx + ux * s))] = 1
            ker /= ker.sum()
            pre = col * A[..., None]
            pre = cv2.filter2D(pre, -1, ker); A = cv2.filter2D(A, -1, ker)
            col = pre / np.maximum(A[..., None], 1e-4)
            mult = cv2.filter2D(mult, -1, ker)
    A = cv2.GaussianBlur(A, (0, 0), 1.4)
    ys, xs = np.nonzero(A > 0.5)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    return col, A, mult, bbox, fp

# ------------------------------------------------------------------ apartment plan (for the app map)
ROOMS = {
    'HALL': (-0.7, 0.7, -5.5, 0.9), 'LIVING': (0.7, 5.0, -5.5, -1.5), 'KITCHEN': (0.7, 5.0, -1.5, 1.6),
    'BEDROOM': (-5.0, -0.7, -2.5, 1.6), 'BATH': (-3.0, -0.7, -5.5, -2.5), 'STUDY': (-5.0, -3.0, -5.5, -2.5),
}
DOORS = [(0.7, -3.5), (0.7, -0.3), (-0.7, -0.8), (-0.7, -4.0), (-4.0, -2.5), (0.0, -5.5), (3.0, -1.5)]
FURN = [(-4.6, -2.6, -1.9, 0.9), (3.9, 4.8, -5.2, -2.7), (2.0, 3.3, 0.0, 0.9), (4.4, 4.95, -1.3, 1.5), (1.6, 3.2, -4.9, -4.2)]
def plan_segs():
    segs = set()
    for (x0, x1, z0, z1) in ROOMS.values():
        for a, b in [((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)), ((x1, z1), (x0, z1)), ((x0, z1), (x0, z0))]:
            n = max(1, round(math.dist(a, b) / 0.1))
            pts = [(round(a[0] + (b[0] - a[0]) * i / n, 2), round(a[1] + (b[1] - a[1]) * i / n, 2)) for i in range(n + 1)]
            for p, q in zip(pts, pts[1:]):
                mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                if any(math.dist(mid, dc) < 0.42 for dc in DOORS): continue
                segs.add(tuple(sorted([p, q])))
    return sorted(segs)
SEGS = plan_segs()
SEG_MID = np.array([((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in SEGS])
ALERT_P = (0.35, -0.9)       # where the person stands (doorway next to the robot)

def robot_plan(t):
    # robot on the map: at the doorway, then into the living room
    k = eio(lin(t, 5.0, 13.0))
    pts = [(0.0, -0.5), (0.0, -2.2), (1.4, -3.5), (2.6, -3.9)]
    s = k * 3; i = min(int(s), 2); u = s - i
    return (lerp(pts[i][0], pts[i + 1][0], u), lerp(pts[i][1], pts[i + 1][1], u))

def draw_map(d, ov, box, t, S, reveal_R=None, robot=(0, -0.5), alert=0.0, beam=True, labels=True):
    """top-down plan inside box (2x screen coords). ov = additive glow layer draw."""
    x0, y0, x1, y1 = box
    s = min((x1 - x0 - 40) / 10.0, (y1 - y0 - 40) / 7.1)
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2; zc = (-5.5 + 1.6) / 2
    M = lambda x, z: (cx + x * s, cy - (z - zc) * s)
    R = 99 if reveal_R is None else reveal_R
    for name, (rx0, rx1, rz0, rz1) in ROOMS.items():
        far = max(math.hypot(x - robot[0], z - robot[1]) for x in (rx0, rx1) for z in (rz0, rz1))
        p = sm(lin(R, far * 0.7, far + 0.2))
        if p <= 0: continue
        a, b = M(rx0, rz1), M(rx1, rz0)
        d.rectangle((a[0], a[1], b[0], b[1]), fill=rgba(C8, 0.07 * p))
        if labels and name != 'HALL':
            c = M((rx0 + rx1) / 2, (rz0 + rz1) / 2)
            d.text((c[0], c[1] - 6), name, font=font('Regular', S * 9), fill=rgba(C8, 0.8 * p), anchor='mm')
            d.text((c[0], c[1] + S * 9), f'{(rx1 - rx0) * (rz1 - rz0):.1f} m²', font=font('Light', S * 7), fill=rgba(C8, 0.5 * p), anchor='mm')
    dist = np.hypot(SEG_MID[:, 0] - robot[0], SEG_MID[:, 1] - robot[1])
    for (p, q), dd in zip(SEGS, dist):
        if dd > R: continue
        b = 0.75 + 0.6 * math.exp(-(R - dd) / 0.5) if reveal_R is not None else 0.85
        ov.line([M(*p), M(*q)], fill=rgba(C8, clamp(b)), width=max(2, int(S * 1.4)))
    for (fx0, fx1, fz0, fz1) in FURN:
        if math.hypot((fx0 + fx1) / 2 - robot[0], (fz0 + fz1) / 2 - robot[1]) > R: continue
        a, b = M(fx0, fz1), M(fx1, fz0)
        d.rectangle((a[0], a[1], b[0], b[1]), outline=rgba(C8, 0.35), width=max(1, int(S * 0.6)))
    rp = M(*robot)
    if beam:
        th = t * 2 * math.pi * 1.2
        Rb = min(R, 4.5) * s
        for k in range(12):
            a0 = th - (k + 1) * 0.06; a1 = th - k * 0.06
            ov.polygon([rp, (rp[0] + math.cos(a0) * Rb, rp[1] + math.sin(a0) * Rb), (rp[0] + math.cos(a1) * Rb, rp[1] + math.sin(a1) * Rb)],
                       fill=rgba(C8, 0.30 * (1 - k / 12) ** 1.5))
        ov.line([rp, (rp[0] + math.cos(th) * Rb, rp[1] + math.sin(th) * Rb)], fill=rgba(C8, 0.9), width=max(2, int(S)))
        if reveal_R is not None and R < 12:
            rr = R * s
            ov.ellipse((rp[0] - rr, rp[1] - rr, rp[0] + rr, rp[1] + rr), outline=rgba(C8, 0.35), width=2)
    r = S * 6
    d.ellipse((rp[0] - r, rp[1] - r, rp[0] + r, rp[1] + r), fill=(8, 12, 18, 255), outline=rgba(C8, 1), width=max(2, int(S * 1.2)))
    d.ellipse((rp[0] - r * 0.35, rp[1] - r * 0.35, rp[0] + r * 0.35, rp[1] + r * 0.35), fill=rgba(C8, 1))
    if alert > 0:
        ap = M(*ALERT_P)
        ph = (t * 1.3) % 1
        rr = S * (5 + 16 * ph)
        d.ellipse((ap[0] - rr, ap[1] - rr, ap[0] + rr, ap[1] + rr), outline=rgba(R8, alert * (1 - ph)), width=max(2, int(S)))
        rr = S * 5
        d.ellipse((ap[0] - rr, ap[1] - rr, ap[0] + rr, ap[1] + rr), fill=rgba(R8, alert))
        ov.ellipse((ap[0] - rr, ap[1] - rr, ap[0] + rr, ap[1] + rr), fill=rgba(R8, alert * 0.8))
        d.text((ap[0] + S * 10, ap[1] - S * 9), 'PERSON', font=font('Medium', S * 8), fill=rgba(R8, alert), anchor='lm')
    return M

# ------------------------------------------------------------------ IR camera tile (from the real footage)
THERM = np.array([[0, 20, 0, 45], [.35, 150, 20, 95], [.6, 255, 70, 40], [.8, 255, 170, 40], [1, 255, 248, 210]], np.float32)
def thermal(v):
    v = np.clip(v, 0, 1)
    return np.stack([np.interp(v, THERM[:, 0], THERM[:, k]) for k in (1, 2, 3)], -1) / 255

def ir_tile(frame, alpha_mask, box_src, size, fi, det, edge_on):
    """frame float 720x1280x3, crop box_src (x0,y0,x1,y1) -> IR-looking tile (w,h) float"""
    w, h = size
    x0, y0, x1, y1 = box_src
    Mx = cv2.getAffineTransform(np.float32([[x0, y0], [x1, y0], [x0, y1]]), np.float32([[0, 0], [w, 0], [0, h]]))
    crop = cv2.warpAffine(frame, Mx, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    l = crop @ np.array([.3, .59, .11], np.float32)
    l = np.clip((l - 0.12) * 1.25, 0, 1) ** 0.9
    rng = np.random.default_rng(fi)
    l = l + rng.standard_normal(l.shape).astype(np.float32) * 0.035
    img = l[..., None] * np.array([0.78, 0.92, 1.0], np.float32)
    yy_, xx_ = np.mgrid[0:h, 0:w].astype(np.float32)
    img *= (1 - 0.55 * (((xx_ / w - .5) ** 2) + ((yy_ / h - .5) ** 2)))[..., None]
    img[::3] *= 0.92
    if alpha_mask is not None and edge_on > 0:
        m = cv2.warpAffine(alpha_mask, Mx, (w, h), flags=cv2.INTER_AREA)
        er = cv2.erode((m > 0.5).astype(np.float32), np.ones((5, 5), np.uint8))
        edge = np.clip((m > 0.5).astype(np.float32) - er, 0, 1)
        glow = cv2.GaussianBlur(edge, (0, 0), 4)
        rel = np.clip(yy_ / h, 0, 1)
        heat = thermal(0.95 - 0.5 * rel)
        img = img * (1 - 0.25 * edge_on * m[..., None]) + thermal(0.5 - 0.25 * rel) * (0.14 * edge_on * m)[..., None]
        img = img + heat * (edge * 0.9 + glow * 1.4)[..., None] * edge_on
    return np.clip(img, 0, 1), Mx

# ------------------------------------------------------------------ phone
SCR_W, SCR_H = 412, 896          # on-screen pixel size of the phone screen
SS = 2                           # supersampling
PW, PH = SCR_W * SS, SCR_H * SS
PHONE_X, PHONE_Y = 1440, 92      # screen top-left in the final frame
VID_X, VID_Y = 72, 180

def rmask(w, h, r):
    m = Image.new('L', (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    return np.asarray(m, np.float32)[..., None] / 255

def paste(dst, src, x, y, mask=None):
    h, w = src.shape[:2]
    x, y = int(x), int(y)
    xa, ya = max(0, x), max(0, y); xb, yb = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if xb <= xa or yb <= ya: return
    s = src[ya - y:yb - y, xa - x:xb - x]
    if mask is None: dst[ya:yb, xa:xb] = s
    else:
        m = mask[ya - y:yb - y, xa - x:xb - x]; dst[ya:yb, xa:xb] = dst[ya:yb, xa:xb] * (1 - m) + s * m

def comp(base, layer):
    a = np.asarray(layer, np.float32) / 255
    return base * (1 - a[..., 3:4]) + a[..., :3] * a[..., 3:4]

def addglow(base, layer, r=6, k=0.9):
    a = np.asarray(layer, np.float32) / 255
    rgb = a[..., :3] * a[..., 3:4]
    g = cv2.GaussianBlur(rgb, (0, 0), r)
    return base * (1 - a[..., 3:4]) + rgb + g * k

def status_bar(d, S):
    # screen-recording red pill around the clock
    d.rounded_rectangle((26 * S, 16 * S, 92 * S, 42 * S), 13 * S, fill=(235, 64, 58, 255))
    d.text((59 * S, 29 * S), '2:14', font=font('SemiBold', 15 * S), fill=(255, 255, 255, 255), anchor='mm')
    d.rounded_rectangle((SCR_W * S / 2 - 62 * S, 11 * S, SCR_W * S / 2 + 62 * S, 47 * S), 18 * S, fill=(0, 0, 0, 255))
    x = (SCR_W - 30) * S
    d.rounded_rectangle((x - 26 * S, 22 * S, x, 36 * S), 4 * S, outline=(235, 240, 245, 200), width=S)
    d.rectangle((x - 24 * S, 24 * S, x - 8 * S, 34 * S), fill=(235, 240, 245, 230))
    for i in range(3):
        d.rectangle((x - 70 * S + i * 7 * S, (34 - 4 - i * 3) * S, x - 66 * S + i * 7 * S, 34 * S), fill=(235, 240, 245, 230))

def phone_screen(t, fi, frame_f, legs, snap):
    S = SS
    scr = np.zeros((PH, PW, 3), np.float32) + np.array([0.025, 0.035, 0.05], np.float32)
    # ---------------- main patrol screen (0 .. ~4.2s), then alert detail slides over it
    detail = eio(lin(t, 3.95, 4.45))
    ui = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
    ov_img = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); ov = ImageDraw.Draw(ov_img)
    off = -60 * S * detail
    d.text((22 * S, 92 * S + off), 'Patrol', font=font('Medium', 32 * S), fill=rgba(WH, 1), anchor='lm')
    d.text((22 * S, 122 * S + off), 'Home · Unit 01 · night routine', font=font('Regular', 12 * S), fill=rgba((150, 160, 172), 1), anchor='lm')
    # mode pill
    blink = 0.4 + 0.6 * (0.5 + 0.5 * math.cos(t * 2 * math.pi * 1.1))
    pa = sm(lin(t, 0.05, 0.4))
    d.rounded_rectangle((22 * S, 140 * S + off, (SCR_W - 22) * S, 176 * S + off), 18 * S, outline=rgba(C8, 0.7 * pa), width=S, fill=rgba(C8, 0.07 * pa))
    d.ellipse((38 * S, 153 * S + off, 48 * S, 163 * S + off), fill=rgba(R8, blink * pa))
    title = 'PATROL MODE · 02:14'
    n = int(len(title) * eo(lin(t, 0.15, 0.9)))
    d.text((58 * S, 158 * S + off), title[:n], font=font('Regular', 14 * S), fill=rgba(C8, pa), anchor='lm')
    d.text(((SCR_W - 38) * S, 158 * S + off), 'ACTIVE', font=font('Medium', 11 * S), fill=rgba(C8, 0.8 * pa), anchor='rm')
    # live map card
    mb = (14 * S, 190 * S + off, (SCR_W - 14) * S, 470 * S + off)
    d.rounded_rectangle(mb, 18 * S, fill=(10, 15, 22, 255), outline=(255, 255, 255, 22), width=S)
    d.text((mb[0] + 14 * S, mb[1] + 18 * S), 'LIVE MAP · LIDAR', font=font('Medium', 10 * S), fill=rgba(C8, 0.9), anchor='lm')
    R = 0.3 + 10.5 * eio(lin(t, 0.3, 3.2))
    det = sm(lin(t, 1.25, 1.45))
    rooms = sum(1 for (x0, x1, z0, z1) in ROOMS.values()
                if R > max(math.hypot(x, z + 0.5) for x in (x0, x1) for z in (z0, z1)) * 0.7 + 0.5)
    d.text((mb[2] - 14 * S, mb[1] + 18 * S), f'MAP {min(100, int(R / 10.8 * 100)):3d}% · {rooms}/6', font=font('Regular', 10 * S), fill=rgba(C8, 0.7), anchor='rm')
    draw_map(d, ov, (mb[0] + 6 * S, mb[1] + 30 * S, mb[2] - 6 * S, mb[3] - 6 * S), t, S, reveal_R=R, robot=(0.0, -0.5), alert=det)
    # camera card
    cb = (14 * S, 484 * S + off, (SCR_W - 14) * S, 760 * S + off)
    d.rounded_rectangle(cb, 18 * S, fill=(10, 15, 22, 255), outline=(255, 255, 255, 22), width=S)
    d.text((cb[0] + 14 * S, cb[1] + 18 * S), 'ROBOT CAM · IR', font=font('Medium', 10 * S), fill=rgba(C8, 0.9), anchor='lm')
    d.text((cb[2] - 14 * S, cb[1] + 18 * S), f'02:14:{7 + int(t):02d}', font=font('Regular', 10 * S), fill=rgba(C8, 0.7), anchor='rm')
    # tab bar
    for i, lab in enumerate(['Map', 'Cam', 'Alerts', 'Unit']):
        x = (SCR_W / 8 + i * SCR_W / 4) * S
        c = C8 if i == 0 else (130, 140, 150)
        d.ellipse((x - 9 * S, 800 * S + off, x + 9 * S, 818 * S + off), outline=rgba(c, 1), width=S)
        d.text((x, 832 * S + off), lab, font=font('Regular', 10 * S), fill=rgba(c, 1), anchor='mm')
    scr = comp(scr, ui)
    scr = addglow(scr, ov_img, r=5 * S, k=0.5)
    # live IR feed in the camera card
    tw, th = int(cb[2] - cb[0] - 20 * S), int(cb[3] - cb[1] - 46 * S)
    ar = tw / th
    if legs is not None and legs['bbox'] is not None:
        bx0, by0, bx1, by1 = legs['bbox']
        fx, fy = (bx0 + bx1) / 2, clamp((by0 + by1) / 2, 200, 620)
    else:
        fx, fy = ROB[min(fi, len(ROB) - 1)]
    fx = clamp(fx, 300, 980)
    cw = 900; ch = cw / ar
    box_src = (fx - cw / 2, fy - ch / 2, fx + cw / 2, fy + ch / 2)
    tile, Mx = ir_tile(frame_f, legs['A'] if legs else None, box_src, (tw, th), fi, det, det)
    tm = rmask(tw, th, 12 * S)
    paste(scr, tile, cb[0] + 10 * S, cb[1] + 36 * S, tm)
    ui = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
    if legs is not None and legs['bbox'] is not None and t > 1.15:
        bx0, by0, bx1, by1 = legs['bbox']
        p0 = Mx @ np.array([bx0, by0, 1.]); p1 = Mx @ np.array([bx1, by1, 1.])
        ox, oy = cb[0] + 10 * S, cb[1] + 36 * S
        x0_, y0_ = max(p0[0], 6 * S) + ox, max(p0[1], 6 * S) + oy
        x1_, y1_ = min(p1[0], tw - 6 * S) + ox, min(p1[1], th - 6 * S) + oy
        p = lin(t, 1.15, 1.4)
        k = lerp(1.5, 1.0, eback(p)); cx_, cy_ = (x0_ + x1_) / 2, (y0_ + y1_) / 2
        hw_, hh_ = (x1_ - x0_) / 2 * k, (y1_ - y0_) / 2 * k
        x0_, x1_, y0_, y1_ = cx_ - hw_, cx_ + hw_, cy_ - hh_, cy_ + hh_
        L = min(x1_ - x0_, y1_ - y0_) * 0.18
        for (x, y, sx, sy) in [(x0_, y0_, 1, 1), (x1_, y0_, -1, 1), (x0_, y1_, 1, -1), (x1_, y1_, -1, -1)]:
            d.line([(x, y + sy * L), (x, y), (x + sx * L, y)], fill=rgba(R8, sm(p * 2)), width=2 * S)
        if p > 0.5:
            conf = int(lerp(71, 94, eo(lin(t, 1.3, 1.65))))
            txt = f'PERSON · {conf}%'
            f = font('Medium', 11 * S); tw_ = f.getlength(txt)
            ty = max(y0_ - 24 * S, cb[1] + 40 * S)
            d.rectangle((x0_, ty, x0_ + tw_ + 14 * S, ty + 20 * S), fill=rgba(R8, 0.92))
            d.text((x0_ + 7 * S, ty + 10 * S), txt, font=f, fill=(12, 12, 16, 255), anchor='lm')
    scr = comp(scr, ui)
    if snap.get('img') is None and t >= 1.7 and legs is not None:
        snap['img'] = tile.copy(); snap['ui'] = scr[int(cb[1] + 36 * S):int(cb[1] + 36 * S) + th, int(cb[0] + 10 * S):int(cb[0] + 10 * S) + tw].copy()

    # ---------------- alert detail screen
    if detail > 0:
        yoff = (1 - detail) * PH
        det_scr = np.zeros((PH, PW, 3), np.float32) + np.array([0.025, 0.035, 0.05], np.float32)
        ui = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
        ov_img = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); ov = ImageDraw.Draw(ov_img)
        d.text((22 * S, 76 * S), '‹ Alerts', font=font('Regular', 14 * S), fill=rgba(C8, 1), anchor='lm')
        d.text((22 * S, 112 * S), 'Motion detected', font=font('Medium', 24 * S), fill=rgba(WH, 1), anchor='lm')
        d.text((22 * S, 140 * S), 'Living room · 02:14:09', font=font('Regular', 13 * S), fill=rgba((170, 180, 190), 1), anchor='lm')
        d.rounded_rectangle((22 * S, 156 * S, 22 * S + font('Medium', 10 * S).getlength('PERSON · 94%') + 16 * S, 178 * S), 6 * S, fill=rgba(R8, 0.95))
        d.text((30 * S, 167 * S), 'PERSON · 94%', font=font('Medium', 10 * S), fill=(12, 12, 16, 255), anchor='lm')
        # map
        mb = (14 * S, 470 * S, (SCR_W - 14) * S, 740 * S)
        d.rounded_rectangle(mb, 18 * S, fill=(10, 15, 22, 255), outline=(255, 255, 255, 22), width=S)
        d.text((mb[0] + 14 * S, mb[1] + 18 * S), 'LIVE MAP', font=font('Medium', 10 * S), fill=rgba(C8, 0.9), anchor='lm')
        rp = robot_plan(t)
        st = 'robot on site · scanning' if t > 11.5 else 'robot en route'
        d.text((mb[2] - 14 * S, mb[1] + 18 * S), st, font=font('Regular', 10 * S), fill=rgba(R8 if t <= 11.5 else C8, 0.85), anchor='rm')
        draw_map(d, ov, (mb[0] + 6 * S, mb[1] + 30 * S, mb[2] - 6 * S, mb[3] - 6 * S), t, S, reveal_R=None, robot=rp, alert=1.0, beam=True)
        # buttons
        by = 760 * S
        for (xa, xb, txt, c) in ((14 * S, (SCR_W / 2 - 6) * S, 'Live view', C8), ((SCR_W / 2 + 6) * S, (SCR_W - 14) * S, 'Sound alarm', R8)):
            d.rounded_rectangle((xa, by, xb, by + 50 * S), 16 * S, outline=rgba(c, 0.85), width=S, fill=rgba(c, 0.10))
            d.text(((xa + xb) / 2, by + 25 * S), txt, font=font('Medium', 14 * S), fill=rgba(c, 1), anchor='mm')
        det_scr = comp(det_scr, ui)
        det_scr = addglow(det_scr, ov_img, r=5 * S, k=0.5)
        status = Image.new('RGBA', (PW, PH), (0, 0, 0, 0))
        if snap.get('ui') is not None:
            si = snap['ui']
            sw = (SCR_W - 28) * S; sh_ = int(sw * si.shape[0] / si.shape[1])
            sh_ = min(sh_, 270 * S)
            big = cv2.resize(si, (int(sw), int(sh_)), interpolation=cv2.INTER_CUBIC)
            paste(det_scr, big, 14 * S, 190 * S, rmask(int(sw), int(sh_), 14 * S))
        canvas = scr * 0.6
        y = int(yoff)
        canvas[y:] = det_scr[:PH - y] if y < PH else canvas[y:]
        scr = canvas if detail < 1 else det_scr

    # ---------------- iOS banner + tap
    bn = eback(lin(t, 2.3, 2.7)) * (1 - eio(lin(t, 3.9, 4.25)))
    if bn > 0:
        by0 = lerp(-110, 54, bn) * S
        bw = (SCR_W - 16) * S; bh = 82 * S
        frost = cv2.GaussianBlur(scr, (0, 0), 18 * S)
        reg = frost * 0.55 + np.array([0.16, 0.17, 0.2], np.float32)
        m = np.zeros((PH, PW, 1), np.float32)
        mm = Image.new('L', (PW, PH), 0); ImageDraw.Draw(mm).rounded_rectangle((8 * S, by0, 8 * S + bw, by0 + bh), 22 * S, fill=255)
        m = np.asarray(mm, np.float32)[..., None] / 255
        press = sm(lin(t, 3.45, 3.55)) * (1 - sm(lin(t, 3.6, 3.75)))
        scr = scr * (1 - m) + reg * (1 - 0.15 * press) * m
        ui = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
        ix, iy = 22 * S, by0 + 14 * S
        d.rounded_rectangle((ix, iy, ix + 34 * S, iy + 34 * S), 9 * S, fill=(8, 14, 22, 255))
        d.ellipse((ix + 8 * S, iy + 8 * S, ix + 26 * S, iy + 26 * S), outline=rgba(C8, 1), width=S * 2)
        d.ellipse((ix + 14 * S, iy + 14 * S, ix + 20 * S, iy + 20 * S), fill=rgba(C8, 1))
        d.text((ix + 46 * S, iy + 8 * S), 'PATROL', font=font('Medium', 11 * S), fill=(215, 222, 230, 230), anchor='lm')
        d.text(((SCR_W - 70) * S, iy + 8 * S), 'now', font=font('Regular', 11 * S), fill=(190, 198, 208, 190), anchor='rm')
        d.text((ix + 46 * S, iy + 29 * S), 'Motion detected', font=font('Medium', 14 * S), fill=(248, 250, 252, 255), anchor='lm')
        d.text((ix + 46 * S, iy + 48 * S), 'Living room · 02:14', font=font('Regular', 13 * S), fill=(220, 226, 234, 235), anchor='lm')
        scr = comp(scr, ui)
        if snap.get('img') is not None:
            th_ = 54 * S; si = snap['img']
            cx_ = si.shape[1] // 2; cy_ = si.shape[0] // 2; half = min(cx_, cy_)
            sq = cv2.resize(si[cy_ - half:cy_ + half, cx_ - half:cx_ + half], (th_, th_), interpolation=cv2.INTER_AREA)
            paste(scr, sq, (SCR_W - 66) * S, iy + 4 * S, rmask(th_, th_, 8 * S))
    # touch indicator
    ta = sm(lin(t, 3.3, 3.42)) * (1 - sm(lin(t, 3.6, 3.8)))
    rp_ = lin(t, 3.55, 4.0)
    if ta > 0 or 0 < rp_ < 1:
        ui = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
        tx, ty = 180 * S, 100 * S
        r = 22 * S * (1 - 0.15 * sm(lin(t, 3.45, 3.55)))
        if ta > 0: d.ellipse((tx - r, ty - r, tx + r, ty + r), fill=(235, 242, 250, int(110 * ta)), outline=(255, 255, 255, int(190 * ta)), width=S)
        if 0 < rp_ < 1:
            rr = (22 + 60 * eo(rp_)) * S
            d.ellipse((tx - rr, ty - rr, tx + rr, ty + rr), outline=(255, 255, 255, int(170 * (1 - rp_))), width=S)
        scr = comp(scr, ui)
    ui = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); status_bar(ImageDraw.Draw(ui), S)
    scr = comp(scr, ui)
    home = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); ImageDraw.Draw(home).rounded_rectangle((PW / 2 - 68 * S, PH - 14 * S, PW / 2 + 68 * S, PH - 9 * S), 3 * S, fill=(235, 240, 245, 220))
    scr = comp(scr, home)
    return cv2.resize(scr, (SCR_W, SCR_H), interpolation=cv2.INTER_AREA)

# ------------------------------------------------------------------ static layers
def phone_body():
    pad = 13
    w, h = SCR_W + 2 * pad, SCR_H + 2 * pad
    im = Image.new('RGBA', (w + 40, h + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((20, 20, 20 + w, 20 + h), 66, fill=(58, 60, 64, 255))
    d.rounded_rectangle((22, 22, 18 + w, 18 + h), 64, fill=(14, 15, 17, 255))
    d.rounded_rectangle((20, 20, 20 + w, 20 + h), 66, outline=(120, 124, 130, 255), width=2)
    for (y0, y1) in ((180, 230), (250, 320), (340, 410)):
        d.rounded_rectangle((16, y0, 21, y1), 2, fill=(70, 72, 76, 255))
    d.rounded_rectangle((19 + w, 270, 24 + w, 380), 2, fill=(70, 72, 76, 255))
    return im, (PHONE_X - pad - 20, PHONE_Y - pad - 20)
BODY, BODY_POS = phone_body()
SCR_MASK = rmask(SCR_W, SCR_H, 54)

def make_label():
    f = font('Medium', 22); txt = 'CONCEPT · SIMULATION'
    pw, ph = int(f.getlength(txt) + 32), 44
    im = Image.new('RGBA', (pw, ph), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, pw - 1, ph - 1), 6, fill=(6, 10, 16, 170), outline=(235, 242, 248, 200), width=2)
    d.text((16, ph / 2), txt, font=f, fill=(240, 246, 250, 240), anchor='lm')
    return im
LABEL = make_label()
VID_MASK = rmask(1280, 720, 16)

def background(frame_f):
    small = cv2.resize(frame_f, (192, 108), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR)
    return bg * 0.22 + np.array([0.02, 0.025, 0.035], np.float32)

def end_card(out, t):
    a = sm(lin(t, 12.2, 12.9))
    if a <= 0: return out
    out = out * (1 - 0.72 * a)
    ui = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
    rise = 18 * (1 - eo(lin(t, 12.2, 13.1)))
    d.text((W / 2, H / 2 - 36 + rise), 'Your vacuum already', font=font('Light', 72), fill=(240, 246, 252, int(255 * a)), anchor='mm')
    d.text((W / 2, H / 2 + 56 + rise), 'knows your home.', font=font('Light', 72), fill=(240, 246, 252, int(255 * a)), anchor='mm')
    lw = 180 * eo(lin(t, 12.7, 13.3))
    d.line([(W / 2 - lw, H / 2 + 132), (W / 2 + lw, H / 2 + 132)], fill=(90, 228, 255, int(200 * a)), width=2)
    return comp(out, ui)

# ------------------------------------------------------------------ main
SLOW_N = 66          # 0.0-1.1 s of the source at 0.5x (the doorway shot with the person)
CUT_SRC = 4.0        # then the patrol continues from source 4.0 s at normal speed
def _raw(args):
    p = subprocess.Popen(['ffmpeg', '-v', 'error'] + args + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(1280 * 720 * 3)
        if len(b) < 1280 * 720 * 3: break
        yield np.frombuffer(b, np.uint8).reshape(720, 1280, 3)
    p.stdout.close(); p.wait()
def read_frames():
    slow = '/tmp/patrol_slow_a.mp4'
    if not os.path.exists(slow):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '0', '-t', '1.2', '-i', SRC, '-an', '-vf',
                        'setpts=2*PTS,minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',
                        '-c:v', 'libx264', '-crf', '10', '-preset', 'fast', slow], check=True)
    k = 0
    for fr in _raw(['-i', slow, '-frames:v', str(SLOW_N)]):
        yield fr, k * 0.5; k += 1
    for j, fr in enumerate(_raw(['-ss', f'{CUT_SRC}', '-i', SRC, '-frames:v', str(N - SLOW_N)])):
        yield fr, CUT_SRC * 30 + j

def main():
    if DEBUG is None:
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                                '-i', SRC, '-map', '0:v', '-map', '1:a?', '-shortest',
                                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-maxrate', '16M', '-bufsize', '32M', '-pix_fmt', 'yuv420p',
                                '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
    else:
        os.makedirs(OUT, exist_ok=True); enc = None
    want = set(DEBUG or range(N)); last = max(want)
    snap = {}; prev = None
    for fi, (fr, sf) in enumerate(read_frames()):
        if fi > last: break
        t = fi / FPS
        need = fi in want or (DEBUG is not None and fi in range(51, 52))   # snapshot frame
        frame_f = fr.astype(np.float32) / 255
        lg = person_layer(sf, prev) if fi < SLOW_N else None
        if lg is not None:
            col, A, mult, bbox, fp = lg; prev = fp
            frame_f = frame_f * mult[..., None]
            frame_f = frame_f * (1 - A[..., None]) + col * A[..., None]
            frame_f = np.clip(frame_f + np.random.default_rng(fi).standard_normal((720, 1280, 1)).astype(np.float32) * 0.012 * A[..., None], 0, 1)
            legs = {'A': A, 'bbox': bbox}
        else:
            legs = None
        if not need: continue
        out = background(frame_f)
        paste(out, frame_f, VID_X, VID_Y, VID_MASK)
        # thin frame around the video
        scr = phone_screen(t, int(sf), frame_f, legs, snap)
        bodyf = np.asarray(BODY, np.float32) / 255
        paste(out, bodyf[..., :3], BODY_POS[0], BODY_POS[1], bodyf[..., 3:4])
        paste(out, scr, PHONE_X, PHONE_Y, SCR_MASK)
        ui = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ui)
        d.rounded_rectangle((VID_X - 1, VID_Y - 1, VID_X + 1280, VID_Y + 720), 16, outline=(255, 255, 255, 40), width=1)
        d.text((VID_X, VID_Y - 30), 'HALLWAY · 02:14', font=font('Regular', 20), fill=(220, 228, 236, 150), anchor='lm')
        out = comp(out, ui)
        out = end_card(out, t)
        img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
        img.paste(LABEL, (W - 40 - LABEL.width, 22), LABEL)
        if enc: enc.stdin.write(img.tobytes())
        else: img.save(f'{OUT}/f{fi:03d}.png')
        if fi % 30 == 0: print('frame', fi, flush=True)
    if enc: enc.stdin.close(); enc.wait()

if __name__ == '__main__':
    main()
