# per-frame camera pose from floor homographies (ref frame 30, assumed intrinsics + ref pose)
import cv2, numpy as np, json, math, sys
F=1900.; K=np.array([[F,0,640],[0,F,360],[0,0,1]],np.float64)
H0=1.50; TH=math.radians(55)
Hs={int(k):np.array(v) for k,v in json.load(open(sys.argv[1])).items()}
# ref camera: world y up, z forward(horizontal). camera axes: x right, y down, z forward
def Rref():
    c,s=math.cos(TH),math.sin(TH)
    # world->cam: x_c = x ; y_c(down) ; z_c(forward)
    return np.array([[1,0,0],[0,-c,s],[0,-s,-c]])*np.array([[1],[1],[1]]) 
R0=np.array([[1,0,0],[0,-math.cos(TH)*0-0,0],[0,0,0]])
c,s=math.cos(TH),math.sin(TH)
# forward f=(0,-s,c), down d = (0,-c,-s), right r=(1,0,0)
R0=np.array([[-1,0,0],[0,-c,-s],[0,-s,c]])
C0=np.array([0,H0,0.])
def backproject(u,v):
    ray=np.linalg.inv(K)@np.array([u,v,1.]); dw=R0.T@ray
    t=-C0[1]/dw[1]; return C0+t*dw
pts_img=[(u,v) for u in range(150,1150,100) for v in range(420,720,60)]
X=np.array([backproject(u,v) for u,v in pts_img])
poses={}
for i,Hm in sorted(Hs.items()):
    p=np.array([[*(Hm@np.array([u,v,1.]))] for u,v in pts_img]); p=p[:,:2]/p[:,2:]
    ok,rv,tv=cv2.solvePnP(X,p,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
    poses[i]=(rv.ravel().tolist(),tv.ravel().tolist())
json.dump(poses,open(sys.argv[2],'w'))
feet=backproject(520,650); print('feet world',feet)
for i in range(0,330,30):
    rv,tv=poses[i]; R,_=cv2.Rodrigues(np.array(rv)); C=-R.T@np.array(tv); print(i, np.round(C,3))
