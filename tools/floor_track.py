import cv2, numpy as np, json, sys
src=sys.argv[1]; REF=30; NF=330
cap=cv2.VideoCapture(src); frames=[]
for i in range(NF):
    ok,f=cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
rob=json.load(open('/tmp/work/track.json'))
def step(a,b,i_a):
    # similarity from frame a to b using features near lower-left floor, excluding robot
    m=np.zeros_like(frames[a]); m[:, :]=255
    p=rob[i_a]
    if p: cv2.circle(m,(int(p[0]-40),int(p[1]-20)),230,0,-1)
    pts=cv2.goodFeaturesToTrack(frames[a],400,0.01,12,mask=m)
    nxt,st,_=cv2.calcOpticalFlowPyrLK(frames[a],frames[b],pts,None,winSize=(31,31),maxLevel=4)
    back,st2,_=cv2.calcOpticalFlowPyrLK(frames[b],frames[a],nxt,None,winSize=(31,31),maxLevel=4)
    good=(st[:,0]==1)&(st2[:,0]==1)&(np.linalg.norm(back-pts,axis=2)[:,0]<1.0)
    H,inl=cv2.findHomography(pts[good],nxt[good],cv2.RANSAC,2.0)
    return H
Hs={REF:np.eye(3)}
for i in range(REF,len(frames)-1): Hs[i+1]=step(i,i+1,i)@Hs[i]
for i in range(REF,0,-1): Hs[i-1]=step(i,i-1,i)@Hs[i]
out={i:Hs[i].tolist() for i in Hs}
json.dump(out,open('/tmp/work2/H.json','w'))
for x,y in [(520,650),(430,560)]:
    for i in range(0,len(frames),15):
        q=Hs[i]@np.array([x,y,1.]); print((x,y),i/30,np.round(q[:2]/q[2]))
