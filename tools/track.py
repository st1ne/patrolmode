import subprocess, numpy as np, json, sys
src=sys.argv[1]
W,H=320,180
p=subprocess.Popen(['ffmpeg','-v','error','-i',src,'-vf',f'scale={W}:{H}','-f','rawvideo','-pix_fmt','gray','-'],stdout=subprocess.PIPE)
out=[]
yy,xx=np.mgrid[0:H,0:W]
while True:
    b=p.stdout.read(W*H)
    if len(b)<W*H: break
    g=np.frombuffer(b,np.uint8).reshape(H,W).astype(np.float32)
    m=g<np.percentile(g,6)+10
    m&=g<62
    if m.sum()<20: out.append(None); continue
    # robust centroid: iterate around median
    cx,cy=np.median(xx[m]),np.median(yy[m])
    for _ in range(3):
        d=(xx-cx)**2+(yy-cy)**2
        mm=m&(d<38**2)
        if mm.sum()<10: break
        cx,cy=xx[mm].mean(),yy[mm].mean()
    out.append((float(cx*4),float(cy*4),int(m.sum())))
json.dump(out,open('/tmp/work/track.json','w'))
for i in range(0,len(out),15): print(i/30,out[i])
