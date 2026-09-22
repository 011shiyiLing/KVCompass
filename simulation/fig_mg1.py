"""fig_mg1 — M/G/1 queueing-delay validation and the reuse/load crossover.

Panel logic (single axes):
  1. Model: Pollaczek-Khinchine E[Tq] = a E[S^2] / (2(1-rho)) with service
     moments measured from the mixed request mix (no caching, S = full
     prefill + first-forward attention).
  2. Validation points: uniform-random forwarding onto single-slot workers
     (random split of a Poisson stream is Poisson per worker, one slot makes
     the worker exactly the M/G/1 server), measured mean queue wait.
  3. Load-aware forwarding without caching (globally least-loaded): same
     offered rho, no reuse, so effective utilization equals offered; its
     knee stays at rho = 1 but delays sit far below random splitting.
  4. Full \sys (score + reuse), offered rho swept PAST 1: reuse shrinks the
     mean residual prefill, so effective utilization is ~0.58x offered and
     the knee moves to offered rho ~ 1/0.58. All curves share the same
     x-axis semantics (offered utilization, no-cache service moments); the
     effective utilization of the full system is reported alongside.
  5. Crossover band: rho*(m) where the model's queueing delay equals the
     prefill saved by an m-token match.

Run:  python3.13 fig_mg1.py            (full)
      python3.13 fig_mg1.py --plot     (replot from cached results json)
Output: ../../figures/mg1_queueing.pdf/png + results/fig_mg1.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGD = HERE.parent.parent / 'figures'
CACHE = HERE / 'results' / 'fig_mg1.json'

import simulate as sim
from simulate import WorkloadGenerator, C_PREFILL_MS_PER_TOKEN, \
    C_DECODE_MS_PER_TOKEN, NUM_NODES

RHOS_MODEL = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
RHOS_LB   = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
RHOS_FULL = [0.30, 0.60, 0.90, 1.10, 1.30, 1.50, 1.65]
N_REQ_A, WARM_A = 150_000, 30_000
N_REQ_A_HI, WARM_A_HI = 600_000, 120_000   # rho >= 0.9: slow convergence
N_REQ_B, WARM_B = 60_000, 12_000
SEED = 42
SVC_PER_TOK = C_PREFILL_MS_PER_TOKEN + C_DECODE_MS_PER_TOKEN


def service_moments(wgen, n=100_000):
    """E[S], E[S^2] of the no-cache service time over the mixed request mix."""
    rng = np.random.default_rng(SEED)
    wgen.reset_stream(SEED)
    s = np.array([wgen.sample(i, 0.0, 'mixed').prefix_len * SVC_PER_TOK
                  for i in range(n)])
    return float(s.mean()), float((s ** 2).mean())


def run_random_lb(wgen, rho, ES):
    """Uniform-random forwarding, single-slot workers, no caching."""
    rng = np.random.default_rng(SEED)
    wgen.reset_stream(SEED)
    n_req, warm = (N_REQ_A_HI, WARM_A_HI) if rho >= 0.9 else (N_REQ_A, WARM_A)
    nodes = [sim.Node(i, 0) for i in range(NUM_NODES)]
    rate_ms = rho * NUM_NODES / ES              # fleet arrivals per ms
    t = 0.0
    for i in range(n_req):
        t += rng.exponential(1.0 / rate_ms)
        req = wgen.sample(i, t, 'mixed')
        nid = int(rng.integers(NUM_NODES))
        nodes[nid].schedule(req.prefix_len, req.prefix_len, t)
    t_warm = warm / rate_ms
    qs = [q for nd in nodes for (arr, q, p) in nd.queue_log if arr > t_warm]
    return float(np.mean(qs))


def _pool_stats(nodes, t_warm):
    """(mean queue wait, effective utilization) over post-warmup entries."""
    qs, ps, t_hi = [], [], 0.0
    for nd in nodes:
        for (arr, q, p) in nd.queue_log:
            if arr > t_warm:
                qs.append(q)
                ps.append(p)
                t_hi = max(t_hi, arr)
    busy = sum(ps) / ((t_hi - t_warm) * NUM_NODES)
    return float(np.mean(qs)), float(busy)


def run_lb_nocache(wgen, rho, ES):
    """Globally least-loaded forwarding, no caching (oracle uniform LB)."""
    from run_section3 import _run_oracle_uniform_lb
    rate_rps = rho * NUM_NODES / ES * 1000.0
    recs, nodes = _run_oracle_uniform_lb(rate_rps, 'mixed', N_REQ_B, WARM_B,
                                         SEED, wgen)
    return _pool_stats(nodes, WARM_B / (rate_rps / 1000.0))


def run_full_sys(wgen, rho, ES):
    """Full \sys: joint score + reuse."""
    from run_section3 import run_scheme
    rate_rps = rho * NUM_NODES / ES * 1000.0
    recs, nodes = run_scheme('ludo', rate_rps, 'mixed', N_REQ_B, WARM_B,
                             SEED, wgen)
    return _pool_stats(nodes, WARM_B / (rate_rps / 1000.0))


def main():
    if '--plot' not in sys.argv:
        sim.MAX_BATCH_SIZE = 1                  # worker == one M/G/1 server
        wgen = WorkloadGenerator(seed=SEED)
        ES, ES2 = service_moments(wgen)
        print(f'E[S]={ES:.1f} ms  E[S^2]={ES2:.0f} ms^2  '
              f'CV^2={(ES2 - ES * ES) / ES ** 2:.2f}')
        rand_pts = []
        for rho in RHOS_MODEL:
            tq = run_random_lb(wgen, rho, ES)
            rand_pts.append((rho, tq))
            print(f'  random-LB rho={rho:.2f}: mean Tq={tq:,.1f} ms', flush=True)
        lb_pts, full_pts = [], []
        for rho in RHOS_LB:
            tq, eff = run_lb_nocache(wgen, rho, ES)
            lb_pts.append((rho, tq, eff))
            print(f'  LB-nocache rho={rho:.2f}: Tq={tq:,.1f} ms '
                  f'(effective rho={eff:.2f})', flush=True)
        for rho in RHOS_FULL:
            tq, eff = run_full_sys(wgen, rho, ES)
            full_pts.append((rho, tq, eff))
            print(f'  full-sys   rho={rho:.2f}: Tq={tq:,.1f} ms '
                  f'(effective rho={eff:.2f})', flush=True)
        CACHE.parent.mkdir(exist_ok=True)
        json.dump({'ES': ES, 'ES2': ES2, 'random': rand_pts,
                   'lb': lb_pts, 'full': full_pts},
                  open(CACHE, 'w'), indent=1)
    D = json.load(open(CACHE))
    ES, ES2 = D['ES'], D['ES2']

    # Crossover band: PK(rho) = beta*m  =>  rho*(m) = bm/(bm + E[S^2]/(2E[S]))
    k = ES2 / (2 * ES)
    m_lo, m_hi = 1024 * C_PREFILL_MS_PER_TOKEN, 4096 * C_PREFILL_MS_PER_TOKEN
    rho_lo = m_lo / (m_lo + k)
    rho_hi = m_hi / (m_hi + k)

    FONT = 'STIXGeneral'
    matplotlib.rcParams.update({'font.family': FONT,
                                'axes.spines.right': False})
    # Honest x-axis: raw arrival rate (req/s), identical semantics for every
    # curve.  rho is shown on a secondary top axis for intuition only.
    RATE0 = NUM_NODES / ES * 1000.0            # arrival rate at rho = 1
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    rr = np.linspace(0.02, 0.985, 400)
    pk = (rr / ES) * ES2 / (2 * (1 - rr))
    ax.plot(rr * RATE0, pk, ls='--', lw=2.8, color='#666666',
            label='M/G/1')
    rx, ry = zip(*D['random'])
    ax.plot(np.array(rx) * RATE0, ry, 'o', ms=9, mfc='white',
            mec='#444444', mew=2.0, label='random')
    lb_pos = [(r, t) for r, t, _ in D['lb'] if t > 0]
    lb_zero = [r for r, t, _ in D['lb'] if t == 0]
    lx, ly = zip(*lb_pos)
    ax.plot(np.array(lx) * RATE0, ly, 's-', lw=3.0, ms=7.2, color='#5A7E9E',
            mfc='#ACBFEA', mec='#5A7E9E', label='load-aware (no reuse)')
    fx, fy, feff = zip(*D['full'])
    ax.plot(np.array(fx) * RATE0, fy, 'o-', lw=3.2, ms=8, color='#985365',
            mfc='#eba79c', mec='#985365', label='KVCompass')
    ax.axvspan(rho_lo * RATE0, rho_hi * RATE0, color='#d9d9d9', alpha=0.5,
               zorder=0)
    for r_ in (rho_lo, rho_hi):
        ax.axvline(r_ * RATE0, color='#777777', lw=1.8, ls=':', ymax=0.68)
    ax.annotate('crossover ' + r'$\rho^{*}$' + '\nm = 1K–4K',
                (0.02, 0.99), xycoords='axes fraction',
                ha='left', va='top', fontsize=26, fontname=FONT,
                color='#555555', linespacing=1.15)
    ax.set_yscale('log')
    ax.set_ylim(6e-3, 3e5)
    ax.set_yticks([1e-2, 1e0, 1e2, 1e4])
    ax.set_xlim(0, 1.75 * RATE0)
    ax.set_xlabel('offered load (requests/s)', fontsize=28, fontname=FONT)
    ax.set_ylabel('mean queueing delay (ms)', fontsize=28, fontname=FONT)
    ax.tick_params(labelsize=25)
    sec = ax.secondary_xaxis('top', functions=(lambda r: r / RATE0,
                                               lambda p: p * RATE0))
    sec.set_xlabel('load relative to no-reuse capacity',
                   fontsize=25, fontname=FONT)
    sec.tick_params(labelsize=23)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.20), ncol=4,
              frameon=False, prop={'family': FONT, 'size': 23},
              handlelength=0.8, columnspacing=0.5, handletextpad=0.25,
              borderaxespad=0.0)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.76, bottom=0.19)
    fig.savefig(FIGD / 'mg1_queueing.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(FIGD / 'mg1_queueing.png', dpi=300, bbox_inches='tight')
    print(f'rho*(1K)={rho_lo:.3f} rho*(4K)={rho_hi:.3f} '
          f'(E[S^2]/2E[S] = {k:.0f} ms)')
    print(f'saved → {FIGD}/mg1_queueing.pdf')


if __name__ == '__main__':
    main()
