"""Effect of torque-command saturation on the Bode curves (Python twin of sat_bode_compare.m).

Assumed channels (GUESS): x1 = excitation/command, x2 = speed feedback, x3 = torque command.
Sweep: log chirp f(t)=f1*(f2/f1)**(t/Tsum); parameters are read from the CSV header.
usage: python sat_bode_compare.py [csv] [torque_limit]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.sans-serif': ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#666', 'axes.labelcolor': '#333', 'text.color': '#222',
                     'xtick.color': '#444', 'ytick.color': '#444', 'grid.color': '#000', 'grid.alpha': .08})
BLUE, ORANGE, RED, GRAY = '#2a6fdb', '#e07b00', '#c0392b', '#888'

csv = sys.argv[1] if len(sys.argv) > 1 else 'ScanFreq_20260928-104750.csv'
NHDR = 36
P = {}
with open(csv) as f:
    for _ in range(NHDR):
        k, *v = f.readline().strip().split(',')
        try: P[k] = float(v[0])
        except (ValueError, IndexError): pass
d = np.loadtxt(csv, delimiter=',', skiprows=NHDR)
x1, x2, x3 = d[:, 0], d[:, 1], d[:, 2]
Fs, Tsum = P.get('Pjet.Fs', 16000), P.get('Pjet.Tsum', 10)
f1, f2 = P.get('Pjet.f1', 5), P.get('Pjet.f2', 5000)
limit = float(sys.argv[2]) if len(sys.argv) > 2 else np.abs(x3).max()
thr = 0.999 * limit

t = np.arange(len(x3)) / Fs
kk = np.log(f2 / f1) / Tsum
phase = 2 * np.pi * f1 * (np.exp(kk * t) - 1) / kk          # instantaneous chirp phase

def window(f0):
    tc = Tsum * np.log(f0 / f1) / np.log(f2 / f1)
    w = min(max(8 / f0, 0.02), 0.3)                          # ~8 cycles, 20-300 ms
    return max(int((tc - w / 2) * Fs), 0), min(int((tc + w / 2) * Fs), len(t))

fq = np.geomspace(6, 4500, 200)
X = np.zeros((len(fq), 3), complex)
sat = np.zeros(len(fq))
for i, f in enumerate(fq):
    a, b = window(f)
    e, w = np.exp(-1j * phase[a:b]), np.hanning(b - a)
    for c, x in enumerate((x1, x2, x3)):
        X[i, c] = (x[a:b] * w * e).sum()
    sat[i] = (np.abs(x3[a:b]) >= thr).mean()

H31, H21, H23 = X[:, 2] / X[:, 0], X[:, 1] / X[:, 0], X[:, 1] / X[:, 2]

# first-order correction: clipped sine -> fundamental gain N(r), r = limit/true amplitude
N = np.ones(len(fq))
m = sat > 0.01
r = np.sin((1 - sat[m]) * np.pi / 2)
N[m] = (2 / np.pi) * (np.arcsin(r) + r * np.sqrt(1 - r ** 2))
Hc31 = H31 / N

db = lambda h: 20 * np.log10(np.abs(h))
err = db(Hc31) - db(H31)
lo, hi = fq[m].min(), fq[m].max()
print(f'Saturated band {lo:.1f}-{hi:.1f} Hz, max clipped {100 * sat.max():.1f} %')
print(f'Max estimated x3/x1 magnitude error {err.max():.2f} dB at {fq[err.argmax()]:.0f} Hz')

def band(ax):
    ax.axvspan(lo, hi, color=RED, alpha=.07, lw=0)
    ax.grid(True, which='both')

fig = plt.figure(figsize=(11, 11.5))
gs = fig.add_gridspec(4, 1, height_ratios=[1, .8, 1.5, .8], hspace=.5)
ax = fig.add_subplot(gs[0])
clip = np.abs(x3) >= thr
ax.plot(t, x3, lw=.4, color=BLUE)
ax.plot(t[clip], x3[clip], '.', ms=1.5, color=RED, label='clipped samples')
ax.axhline(limit, color=RED, ls='--', lw=1); ax.axhline(-limit, color=RED, ls='--', lw=1, label=f'limit ±{limit:.3f}')
ax.set(xlabel='t [s]', ylabel='x3', title='x3 (torque command) in time domain'); ax.legend(loc='upper right')

ax = fig.add_subplot(gs[1]); ax.semilogx(fq, 100 * sat, color=RED, lw=1.5); band(ax)
ax.set(ylabel='clipped %', title='Fraction of x3 samples on the limit, per frequency')

ax = fig.add_subplot(gs[2]); band(ax)
ax.semilogx(fq, db(H31), color=BLUE, lw=1.6, label='x3/x1 measured (with saturation)')
ax.semilogx(fq, db(Hc31), color=ORANGE, lw=1.6, ls='--', label='x3/x1 corrected (describing function)')
ax.set(ylabel='magnitude [dB]', title='x3/x1 magnitude: measured vs corrected'); ax.legend(loc='upper right')
ax.annotate(f'{err.max():.1f} dB', (fq[err.argmax()], db(Hc31)[err.argmax()]), (fq[err.argmax()] * 1.6, db(Hc31)[err.argmax()] - 10),
            arrowprops=dict(arrowstyle='-', color=GRAY), color='#222')

ax = fig.add_subplot(gs[3]); band(ax); ax.semilogx(fq, err, color='#222', lw=1.5)
ax.set(xlabel='Frequency [Hz]', ylabel='dB', title='Estimated magnitude error caused by saturation')
fig.savefig('sat_bode_compare.png', dpi=110, bbox_inches='tight')

# second figure: which curves are affected
fig2, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for j, (H, nm) in enumerate(((H23, 'x2/x3  plant (uses the actual clipped torque)'), (H21, 'x2/x1  closed loop (affected by clipping)'))):
    axs[0, j].semilogx(fq, db(H), color=BLUE, lw=1.5); axs[0, j].set(title=nm, ylabel='magnitude [dB]')
    axs[1, j].semilogx(fq, np.degrees(np.unwrap(np.angle(H))), color=BLUE, lw=1.5); axs[1, j].set(ylabel='phase [deg]', xlabel='Frequency [Hz]')
    band(axs[0, j]); band(axs[1, j])
fig2.suptitle('Shaded: saturated band. Above ~500 Hz x2 is quantisation-limited.', y=.995, fontsize=10)
fig2.tight_layout(); fig2.savefig('sat_bode_curves.png', dpi=110)
