from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from yam_pld_rl.kinematics import YamKinematics
out=Path(__file__).resolve().parent
z=np.load(out/"continuity-signals.npz")
t,actual,pi,delta,mask=(z[k] for k in ("t","actual","pi","delta","mask"))
fk=YamKinematics(Path.cwd()/"third_party/yam-fk-model/yam_linear4310.xml")
plt.rcParams.update({'font.size':9})
fig,axes=plt.subplots(4,1,figsize=(13,9),sharex=True)
for j in range(6):axes[0].plot(t,delta[:,7+j],label=f'J{j+1}',lw=1)
axes[0].set_ylabel('Added residual (rad)');axes[0].legend(ncol=6,loc='upper right');axes[0].set_title('Recorded collect: 13 short residual pulses over two right-arm attempts')
for label,arr,color in [('Pi reference',pi,'#888'),('Submitted Pi + residual',actual,'#df5a39')]:axes[1].plot(t,arr[:,9],label=label,color=color,lw=1.2)
axes[1].set_ylabel('Right J3 target (rad)');axes[1].legend()
xyz=fk.positions(actual[:,7:13])-fk.positions(pi[:,7:13])
for j,lab in enumerate(('dx','dy','dz')):axes[2].plot(t,xyz[:,j]*1000,label=lab)
axes[2].set_ylabel('Added target FK (mm)');axes[2].legend(ncol=3)
axes[3].step(t,mask,label='nonzero residual',where='post');axes[3].set_ylabel('Residual active');axes[3].set_xlabel('Seconds since recording start')
for ax in axes:ax.grid(alpha=.2);ax.set_xlim(1.3,8.2)
fig.tight_layout();fig.savefig(out/'continuity.png',dpi=150);plt.close(fig)
