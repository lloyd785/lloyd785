"""Bode from ScanFreq CSV under guessed roles: x1=excitation/cmd, x2=speed fb, x3=torque cmd.
Log chirp f=5*1000^(t/10); per-frequency lock-in; ideal plant 1/(J s) with 1.5 Ts delay."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','WenQuanYi Zen Hei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
f_csv=sys.argv[1]; d=np.loadtxt(f_csv,delimiter=',',skiprows=36)
fs,T,L=16000,10,0.4998; t=np.arange(len(d))/fs; k=np.log(1000)/T
ph=2*np.pi*5*(np.exp(k*t)-1)/k
def li(x,f0):
    tc=T*np.log(f0/5)/np.log(1000); w=min(max(8/f0,.02),.3)
    a=max(int((tc-w/2)*fs),0); b=min(int((tc+w/2)*fs),len(x))
    return (x[a:b]*np.hanning(b-a)*np.exp(-1j*ph[a:b])).sum(), (np.abs(d[a:b,2])>=L).mean()
fq=np.geomspace(6,4500,200)
X=[np.array([li(d[:,i],f)[0] for f in fq]) for i in range(3)]
sat=np.array([li(d[:,2],f)[1] for f in fq])
P=X[1]/X[2]; T21=X[1]/X[0]; C=X[2]/X[0]
db=lambda h:20*np.log10(abs(h)); pd=lambda h:np.degrees(np.unwrap(np.angle(h)))
w=2*np.pi*fq; m=(fq>=8)&(fq<=25)&(sat<0.01)
c=10**(np.mean(db(P[m])+20*np.log10(w[m]))/20)   # |P|=c/w fitted at 8-25 Hz
Pid=c/(1j*w)*np.exp(-1j*w*1.5/fs)
bad=(fq>500)
fig,ax=plt.subplots(3,2,figsize=(12,10),sharex=True)
for j,(H,name,ideal) in enumerate(((P,'速度反馈/转矩指令(被控对象)',Pid),(T21,'速度反馈/激励(闭环)',None))):
    ax[0,j].semilogx(fq,db(H),label='实测');ax[1,j].semilogx(fq,pd(H))
    if ideal is not None:
        ax[0,j].semilogx(fq,db(ideal),'--',label=f'理想 c/(jω)+1.5Ts 延迟, c={c:.3g}');ax[1,j].semilogx(fq,pd(ideal),'--')
    ax[0,j].set_title(name);ax[0,j].legend();ax[0,j].set_ylabel('幅值 dB');ax[1,j].set_ylabel('相位 °')
    for a in (ax[0,j],ax[1,j]):
        a.axvspan(500,4500,color='gray',alpha=.15);a.grid(True,which='both',alpha=.3)
        for lo,hi in ((fq[sat>0.01].min(),fq[sat>0.01].max()),):a.axvspan(lo,hi,color='r',alpha=.08)
    ax[2,j].semilogx(fq,sat*100,c='C3');ax[2,j].set_ylabel('转矩指令限幅 %');ax[2,j].set_xlabel('频率 Hz');ax[2,j].grid(True,which='both',alpha=.3)
fig.suptitle('灰:速度反馈分辨率不足(>500Hz 不可信)  红:转矩指令饱和(50% 限制)');plt.tight_layout();plt.savefig('bode_guess.png',dpi=105)
print('c',c)
