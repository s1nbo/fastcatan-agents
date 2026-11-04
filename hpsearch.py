"""Hyperparameter search in two stages: equal-hours training runs, compared head to head.

    python hpsearch.py ppo --search ppo-s1 --trials 16 --hours 1     # stage 1: SPACE at the default size
    python hpsearch.py ppo --search ppo-s2 --sizes --hours 2         # stage 2: SIZES on the stage-1 winner
    python hpsearch.py ppo --search ppo-s1 --report                  # only redo report.md and the figures
    python ppo.py --name ppo_best --hours 72 --config results/hpsearch_ppo_best.json

Protocol (the same for ppo, dqn and az):
- Stage 1: trial 00 trains the hand-set defaults, trials 01.. random draws from SPACE (every other
  constant unchanged, the default network size). Stage 2: the stage-1 winner at every size in
  SIZES; trial 00 is the size it already had. P_SELFPLAY is fixed (0.8 in every trainer).
- Every trial: --hours of training, trades on, seed TRIAL_SEED (the same boards and dice for all,
  so differences come from the settings), one trial at a time (equal hours = equal compute).
- Its final checkpoint then plays EVAL_GAMES games on the same boards (EVAL_SEED), 2v2 with the
  seats swapped, against
  * vs_ref:    the reference, trial 00's final checkpoint (trades on): head to head;
  * vs_ab1:    Alpha-Beta K=1 with player trades off for every seat (the benchmark setting);
  * vs_random: random (trades on);
  as a per-seat win rate with a Wilson 95% interval (chance 0.25; truncated games count for
  nobody). Then SELFPLAY_GAMES games against itself with trade logs give its trading: offers,
  trades and acceptance per game, acceptance by the proposer's lead and the lead slope (metrics.py).
- Ranking: by RANK_BY (vs_ref), then vs_ab1, then vs_random. Between agents this close, head to
  head has the most statistical power; after short trials Alpha-Beta beats nearly every agent and
  every agent beats random, so those two separate little. The winner's settings go to
  results/hpsearch_<algo>_best.json (after stage 2: with its size).

Files, all under results/hpsearch_<algo>_<search>/ (the data for the write-up):
- plan.json     the protocol: constants above, SPACE or SIZES, every trial's config, the trainer
                defaults, code commit (code.diff: uncommitted changes), versions and machine;
- trials.jsonl  one row per finished trial: config, all settings, training totals, every
                evaluation (rate, ci95, wins, games, seconds), trading statistics, score;
- results.csv   the same as a flat table (for pgfplots or pandas);
- report.md     protocol, search space and results tables, the winner, figure captions;
- figures/      ranking, sensitivity (stage 1) or sizes and speed (stage 2), learning curves,
                trading; each as .pdf (LaTeX) and .png.
Each trial's run is runs/hp_<algo>_<search>_NN/: config.json, metrics.jsonl (the learning curve),
ckpt_*.pt and the self-play trade logs selfplay.A02.fcev / .A13.fcev. Its pool/ and resume.pt are
deleted after the evaluation to save disk.

A search is resumable: after a crash, rerun the same command (same --search). Finished trials are
skipped and an unfinished one continues from its resume.pt. The command exits non-zero if a trial
failed, so `until python hpsearch.py ...; do sleep 60; done` retries it.

Caveats for the write-up: the trials are short (the real runs take days), which favours settings
that learn fast; one seed per trial, so confirm the winner with more seeds; league snapshots come
every 20k games, so a short trial meets few of them (an AZ trial often none).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy/torch/fastcatan load (see common.py)

import argparse
import csv
import importlib
import itertools
import json
import multiprocessing as mp
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from common import ROOT
from evaluate import load, match, per_seat

MODULES = {"ppo": "ppo", "dqn": "dqn", "az": "alphazero"}
SPACE = {  # ("log", lo, hi) = log-uniform, a tuple of values = one of them
    "ppo": dict(LR=("log", 3e-5, 3e-4), ENT_COEF=("log", 1e-3, 3e-2), GAMMA=(0.995, 0.999, 0.9995),
                LAMBDA=(0.9, 0.95, 0.98), CLIP=(0.1, 0.2, 0.3), EPOCHS=(2, 4, 8), NUM_ENVS=(512, 1024, 2048)),
    "dqn": dict(LR=("log", 1e-5, 3e-4), N_STEP=(3, 5, 10), GAMMA=(0.995, 0.999), EPS_DECAY=(0.01, 0.02, 0.05),
                TARGET_EVERY=(500, 2_000, 5_000), BATCH=(256, 512, 1024), TRAIN_EVERY=(1, 2, 4)),
    "az": dict(N_SIMS=(16, 32, 64), C_PUCT=(1.0, 1.5, 2.5), LR=("log", 1e-4, 1e-3), REPLAY_RATIO=(2, 4, 8),
               TOP_K=(8, 16)),
}
SIZES = dict(WIDTH=(1024, 2048), DEPTH=(3, 5, 7))  # stage 2: every combination, large nets only
EVAL_GAMES, SELFPLAY_GAMES, EVAL_SEED, TRIAL_SEED = 400, 200, 7_000, 1
RANK_BY = "vs_ref"  # the ranking score: "vs_ref" or "vs_ab1"
EVALS = {"vs_ref": "the reference (trial 00), trades on", "vs_ab1": "Alpha-Beta K=1, player trades off",
         "vs_random": "random, trades on"}


def draw(space, rng):
    return {k: float(np.exp(rng.uniform(np.log(v[1]), np.log(v[2])))) if v[0] == "log"
            else v[int(rng.integers(len(v)))] for k, v in space.items()}


def make_plan(algo, search, stage, trials, hours, seed, out):
    """Every trial's config, and the protocol, versions and machine of this search."""
    import fastcatan as fc
    import torch

    module = importlib.import_module(MODULES[algo])
    defaults = {k: v for k, v in vars(module).items() if k.isupper() and isinstance(v, (int, float))}
    if stage == 2:
        best = ROOT / "results" / f"hpsearch_{algo}_best.json"
        if not best.exists():
            raise SystemExit(f"{best} is missing: run stage 1 first")
        base = json.loads(best.read_text())
        first = (base.get("WIDTH", defaults["WIDTH"]), base.get("DEPTH", defaults["DEPTH"]))
        sizes = [first] + [s for s in itertools.product(SIZES["WIDTH"], SIZES["DEPTH"]) if s != first]
        configs = [dict(base, WIDTH=w, DEPTH=d) for w, d in sizes]
    else:
        rng = np.random.default_rng(seed)
        configs = [{}] + [draw(SPACE[algo], rng) for _ in range(trials - 1)]

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout

    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    if dirty:
        (out / "code.diff").write_text(git("diff", "HEAD"))
    return dict(algo=algo, search=search, stage=stage, hours=hours, trials=len(configs), configs=configs,
                space=SIZES if stage == 2 else SPACE[algo], defaults=defaults, rank_by=RANK_BY, evals=EVALS,
                eval_games=EVAL_GAMES, selfplay_games=SELFPLAY_GAMES, eval_seed=EVAL_SEED, trial_seed=TRIAL_SEED,
                draw_seed=seed, created=time.strftime("%Y-%m-%d %H:%M"),
                code=dict(commit=git("rev-parse", "HEAD").strip(), dirty=dirty),
                versions=dict(fastcatan=fc.__version__, torch=torch.__version__, python=platform.python_version()),
                machine=dict(host=platform.node(), cpus=os.cpu_count(),
                             gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))


def trial(algo, search, i, config, hours):
    """Train trial i (or continue it after a crash), evaluate its final checkpoint, append its row."""
    import torch

    from metrics import summarize

    out = ROOT / "results" / f"hpsearch_{algo}_{search}"
    name = f"hp_{algo}_{search}_{i:02d}"
    run = ROOT / "runs" / name
    t = time.time()
    importlib.import_module(MODULES[algo]).train(name, TRIAL_SEED, hours, True, config, resume=True)
    wall = time.time() - t

    def final(run_name):
        return str(sorted((ROOT / "runs" / run_name).glob("ckpt_*.pt"))[-1])

    torch.manual_seed(EVAL_SEED)
    agent = load(final(name), seed=1)
    ref = agent if i == 0 else load(final(f"hp_{algo}_{search}_00"), seed=2)
    evals = {}
    for key, opponent, trades in (("vs_ref", ref, True), ("vs_ab1", load("ab:1", seed=3), False),
                                  ("vs_random", load("random", seed=4), True)):
        t = time.time()
        a_wins, b_wins, truncated, games = match(agent, opponent, EVAL_GAMES, EVAL_SEED, trades, trades)
        rate, ci = per_seat(a_wins, games)
        evals[key] = dict(rate=rate, ci95=list(ci), wins=a_wins, opponent_wins=b_wins, truncated=truncated,
                          games=games, seconds=round(time.time() - t))
    t = time.time()
    for log in run.glob("selfplay.*.fcev"):  # a trial that crashed during its evaluation
        log.unlink()
    match(agent, agent, SELFPLAY_GAMES, EVAL_SEED + 1, events=run / "selfplay.fcev", labels=(name, name))
    stats = summarize(sorted(run.glob("selfplay.A*.fcev")), name)
    trading = {k: stats[k] for k in ("games", "offers_per_game", "trades_per_game", "accept_rate", "accept_by_lead",
                                     "lead_slope", "offers_per_turn_by_lead", "offer_terms_by_lead",
                                     "trade_terms_by_lead", "reciprocity")}
    trading["seconds"] = round(time.time() - t)

    last = [r for r in map(json.loads, open(run / "metrics.jsonl")) if "total_games" in r][-1]
    settings = {k: v for k, v in json.loads((run / "config.json").read_text()).items() if k.isupper()}
    row = dict(algo=algo, search=search, trial=i, name=name, config=config, settings=settings, hours=hours,
               ckpt=final(name), score=evals[RANK_BY]["rate"], eval=evals, trading=trading,
               training=dict(env_steps=last["env_steps"], games=last["total_games"], hours=last["hours"],
                             env_steps_per_s=last["env_steps"] / max(1e-9, last["hours"] * 3600),
                             wall_seconds=round(wall)))
    with open(out / "trials.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    shutil.rmtree(run / "pool", ignore_errors=True)  # not needed any more: save the disk
    (run / "resume.pt").unlink(missing_ok=True)
    e = {k: v["rate"] for k, v in evals.items()}
    print(f"trial {i:02d}: vs ref {e['vs_ref']:.3f}  vs AB {e['vs_ab1']:.3f}  vs random {e['vs_random']:.3f}  "
          f"trades/game {trading['trades_per_game']}  {config}", flush=True)


def finished(out):
    path = out / "trials.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def ranked(rows):
    order = [RANK_BY] + [k for k in EVALS if k != RANK_BY]
    return sorted(rows, key=lambda r: [r["eval"][k]["rate"] for k in order], reverse=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("algo", choices=list(MODULES))
    p.add_argument("--search", help="name of the search (default: a timestamp); rerun with it to continue")
    p.add_argument("--sizes", action="store_true", help="stage 2: every size in SIZES on the stage-1 winner")
    p.add_argument("--trials", type=int, default=16, help="stage 1: trial 00 (the defaults) plus trials-1 draws")
    p.add_argument("--hours", type=float, default=1.0, help="training budget per trial")
    p.add_argument("--parallel", type=int, default=1,
                   help="trials at the same time after trial 00 (they share the GPU: equal hours, unequal compute)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random draws")
    p.add_argument("--report", action="store_true", help="only (re)write report.md and the figures of --search")
    args = p.parse_args()
    if args.report and not args.search:
        p.error("--report needs --search")
    search = args.search or time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / f"hpsearch_{args.algo}_{search}"
    if args.report:
        return report(out)

    stage = 2 if args.sizes else 1
    if (out / "plan.json").exists():
        plan = json.loads((out / "plan.json").read_text())
        if (plan["stage"], plan["hours"]) != (stage, args.hours):
            p.error(f"{out.name} is a stage-{plan['stage']} search with {plan['hours']} h trials")
        print(f"continuing {out.name}: {len(finished(out))} of {plan['trials']} trials done", flush=True)
    else:
        out.mkdir(parents=True)
        plan = make_plan(args.algo, search, stage, args.trials, args.hours, args.seed, out)
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    configs, done = plan["configs"], {r["trial"] for r in finished(out)}
    todo = [i for i in range(len(configs)) if i not in done]
    rest = [i for i in todo if i != 0]
    batches = ([[0]] if 0 in todo else []) + [rest[k:k + args.parallel] for k in range(0, len(rest), args.parallel)]

    ctx, failed = mp.get_context("spawn"), []  # each trial in its own process (AZ trials start their own actors)
    for batch in batches:
        procs = {i: ctx.Process(target=trial, args=(args.algo, search, i, configs[i], plan["hours"])) for i in batch}
        for proc in procs.values():
            proc.start()
        for i, proc in procs.items():
            proc.join()
            if proc.exitcode != 0:
                failed.append(i)
        if 0 in failed:
            break  # every other trial is evaluated against trial 00

    rows = ranked(finished(out))
    for rank, r in enumerate(rows, 1):
        e = r["eval"]
        print(f"{rank:2d}. trial {r['trial']:02d}  vs ref {e['vs_ref']['rate']:.3f}  vs AB {e['vs_ab1']['rate']:.3f}  "
              f"vs random {e['vs_random']['rate']:.3f}  {r['config']}")
    if failed:
        sys.exit(f"trials {failed} failed (tracebacks above); rerun with --search {search} to retry them")
    best = ROOT / "results" / f"hpsearch_{args.algo}_best.json"
    best.write_text(json.dumps(rows[0]["config"], indent=2) + "\n")
    print(f"best settings (trial {rows[0]['trial']:02d}) -> {best}")
    try:
        report(out)
    except Exception as e:  # never fail a finished search over its figures; --report redoes them
        print(f"report failed ({e!r}); fix it, then run with --report", flush=True)


# --- report: tables and figures for the write-up ---------------------------------------------

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated together, light mode)
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue
NAMES = {"vs_ref": "vs reference (trial 00)", "vs_ab1": "vs Alpha-Beta K=1", "vs_random": "vs random",
         "env_steps": "env steps (millions)", "steps_per_turn": "decisions per turn", "pg_loss": "policy loss",
         "p_loss": "policy loss", "v_loss": "value loss", "loss": "TD loss", "entropy": "policy entropy",
         "pool_winrate": "win share vs league"}
CURVES = ("env_steps", "steps_per_turn", "pg_loss", "p_loss", "loss", "v_loss", "entropy", "pool_winrate")


def label(r):
    return f"{r['trial']:02d}" + (" (reference)" if r["trial"] == 0 else "")


def changed(r, plan):
    """The settings of trial r that differ from the trainer defaults, short."""
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["config"].items()
                     if v != plan["defaults"].get(k)) or "defaults"


def interval(e):
    return f"{e['rate']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"


def report(out):
    """report.md, results.csv and figures/ from a search's plan.json and trials.jsonl."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plan, rows = json.loads((out / "plan.json").read_text()), ranked(finished(out))
    if not rows:
        return print(f"{out.name}: no finished trials yet")
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200,
                         "figure.constrained_layout.use": True})
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    key, short = plan["rank_by"], NAMES[plan["rank_by"]].split(" (")[0]
    ref = next((r for r in rows if r["trial"] == 0), None)
    written = []

    def save(fig, name, caption):
        for ext in ("pdf", "png"):
            fig.savefig(figs / f"{name}.{ext}")
        plt.close(fig)
        written.append((name, caption))

    def errorbars(ax, x, y, lo, hi, color, horizontal=False, **kw):
        err = [np.subtract(y, lo), np.subtract(hi, y)]
        xy, bars = ((y, x), {"xerr": err}) if horizontal else ((x, y), {"yerr": err})
        ax.errorbar(*xy, **bars, fmt="o", color=color, ms=4.5, elinewidth=1.4, capsize=0, mec="white", mew=0.8, **kw)

    def chance(ax, vertical=False):
        (ax.axvline if vertical else ax.axhline)(0.25, color=INK_2, lw=0.8, ls=(0, (3, 3)))

    # 1. Ranking: every trial's three evaluations, sorted by the ranking score.
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7, 0.8 + 0.22 * len(rows)), sharey=True)
    for ax, opponent in zip(axes, EVALS):
        e = [r["eval"][opponent] for r in rows]
        errorbars(ax, y, [v["rate"] for v in e], [v["ci95"][0] for v in e], [v["ci95"][1] for v in e], BLUE,
                  horizontal=True)
        chance(ax, vertical=True)
        ax.set(xlim=(0, 0.5), title=NAMES[opponent], xlabel="per-seat win rate")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label(r) for r in rows])
    axes[0].set_ylabel("trial (best first)")
    save(fig, "ranking", f"Per-seat win rate of every trial's final checkpoint after {plan['hours']} h of training "
         f"({plan['eval_games']} games each, 2v2 with seats swapped, Wilson 95% intervals; dashed: chance 0.25). "
         f"Trials are sorted by {NAMES[key]}.")

    score = np.array([r["score"] for r in rows])
    ci = np.array([r["eval"][key]["ci95"] for r in rows])
    if plan["stage"] == 1:  # 2. Sensitivity: the score against each searched setting.
        names = list(plan["space"])
        cols = 4 if len(names) > 6 else 3
        n_rows = -(-len(names) // cols)
        fig, axes = plt.subplots(n_rows, cols, figsize=(7, 1.9 * n_rows), sharey=True, squeeze=False)
        is_ref = np.array([r["trial"] == 0 for r in rows])
        for ax, name in zip(axes.flat, names):
            spec, values = plan["space"][name], [r["settings"][name] for r in rows]
            if spec[0] == "log":
                x = np.array(values, float)
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 3.0)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:  # categories, with a little jitter so equal values stay visible
                cats = sorted(set(map(float, spec)) | {float(plan["defaults"][name])})
                x = np.array([cats.index(float(v)) for v in values]) + np.linspace(-0.12, 0.12, len(rows))
                ax.set_xticks(range(len(cats)), [f"{c:g}" for c in cats])
                ax.set_xlim(-0.5, len(cats) - 0.5)
            errorbars(ax, x[~is_ref], score[~is_ref], ci[~is_ref, 0], ci[~is_ref, 1], BLUE, label="random draw")
            errorbars(ax, x[is_ref], score[is_ref], ci[is_ref, 0], ci[is_ref, 1], ORANGE, label="defaults (trial 00)")
            chance(ax)
            ax.set_title(name)
        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"win rate {short}")
        spare = axes.flat[len(names)] if len(names) < axes.size else None
        if spare is not None:  # the legend in the empty slot, else above the panels
            spare.set_visible(True)
            spare.axis("off")
            spare.legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
        else:
            fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper left", ncol=2)
        save(fig, "sensitivity", f"Ranking score ({NAMES[key]}, per seat, Wilson 95%) against each searched "
             "setting. Orange: the hand-set defaults (trial 00), the reference that every trial plays.")
    else:  # 2. Sizes: score and speed over width x depth.
        widths, depths = sorted(plan["space"]["WIDTH"]), sorted(plan["space"]["DEPTH"])
        panels = (("sizes", lambda r: r["score"], f"win rate {NAMES[key]}",
                   lambda r: f"{r['score']:.3f}\n[{r['eval'][key]['ci95'][0]:.3f}, {r['eval'][key]['ci95'][1]:.3f}]"),
                  ("speed", lambda r: r["training"]["env_steps_per_s"] / 1e3, "env steps per second (thousands)",
                   lambda r: f"{r['training']['env_steps_per_s'] / 1e3:.0f}k"))
        for name, value, title, text in panels:
            grid, cells = np.full((len(depths), len(widths)), np.nan), {}
            for r in rows:
                j, k = depths.index(r["settings"]["DEPTH"]), widths.index(r["settings"]["WIDTH"])
                grid[j, k], cells[j, k] = value(r), text(r)
            fig, ax = plt.subplots(figsize=(1.2 + 1.3 * len(widths), 0.6 + 0.7 * len(depths)))
            norm = Normalize(np.nanmin(grid), np.nanmax(grid) + 1e-9)
            im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")
            for (j, k), s in cells.items():
                ax.text(k, j, s, ha="center", va="center", fontsize=6.5,
                        color="white" if norm(grid[j, k]) > 0.55 else INK)
            ax.set_xticks(range(len(widths)), widths)
            ax.set_yticks(range(len(depths)), depths)
            ax.set(xlabel="width (units per layer)", ylabel="depth (hidden layers)", title=title)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.85)
            save(fig, name, f"{title[0].upper() + title[1:]} after {plan['hours']} h of training, by network size "
                 "(the stage-1 winner's settings).")

    # 3. Learning curves, each trial colored by its final score.
    curves = {r["trial"]: [c for c in map(json.loads, open(ROOT / "runs" / r["name"] / "metrics.jsonl"))
                           if "resumed" not in c] for r in rows}
    metrics = [m for m in CURVES if any(m in c for cs in curves.values() for c in cs)]
    norm = Normalize(score.min(), score.max() + 1e-9)
    cols = min(3, len(metrics))
    fig, axes = plt.subplots(-(-len(metrics) // cols), cols, figsize=(7, 2.0 * -(-len(metrics) // cols)),
                             squeeze=False)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    for ax, metric in zip(axes.flat, metrics):
        for r in sorted(rows, key=lambda r: r["score"]):  # the best drawn last, on top
            pts = [(c["hours"], c[metric]) for c in curves[r["trial"]] if c.get(metric) is not None
                   and (metric != "steps_per_turn" or c.get("games"))]  # no finished game yet: no value
            if not pts:
                continue
            h, v = np.array(pts, float).T
            idx = np.digitize(h, np.linspace(0, h.max(), 41))  # time bins: per-update rows are noisy
            hb = np.array([h[idx == b].mean() for b in np.unique(idx)])
            vb = np.array([v[idx == b].mean() for b in np.unique(idx)]) / (1e6 if metric == "env_steps" else 1)
            ax.plot(hb, vb, color=cmap(norm(r["score"])), lw=1.1)
        ax.set(title=NAMES[metric], xlabel="hours")
    fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=axes, shrink=0.8, label="final score")
    save(fig, "curves", f"Training curves of every trial (time-binned means), colored by its final ranking score "
         f"({NAMES[key]}): darker is better.")

    # 4. Trading in self-play: how much the agents trade, and acceptance by the proposer's lead.
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.3))
    errorbars(a, [r["trading"]["trades_per_game"] for r in rows], score, ci[:, 0], ci[:, 1], BLUE)
    chance(a)
    a.set(xlabel="trades per game in self-play", ylabel=f"win rate {short}", title="trading and strength")
    leads = np.arange(-3, 4)
    shown = [(rows[0], BLUE, -0.08, "best")] + ([(ref, ORANGE, 0.08, "reference")] if ref and ref is not rows[0] else [])
    for r, color, offset, role in shown:
        cells = [r["trading"]["accept_by_lead"].get(str(d)) for d in leads]
        ok = np.array([c is not None for c in cells])
        if ok.any():
            v = np.array([c for c in cells if c is not None])
            b.plot(leads[ok] + offset, v[:, 0], color=color, lw=1, alpha=0.6)
            errorbars(b, leads[ok] + offset, v[:, 0], v[:, 1], v[:, 2], color, label=f"trial {r['trial']:02d} ({role})")
    b.set(xlabel="proposer's lead (VP over the best other seat)", ylabel="acceptance rate", xticks=leads,
          title="willingness to trade with the leader")
    low, high = b.get_ylim()
    b.set_ylim(low, high + 0.3 * (high - low))  # room for the legend above the points
    b.legend(loc="upper left", ncol=2)
    save(fig, "trading", f"Left: ranking score against trades per game ({plan['selfplay_games']} self-play games per "
         "trial). Right: share of offers accepted by the proposer's lead, for the best trial and the reference "
         "(Wilson 95%); a falling line means the leader is frozen out.")

    # results.csv: one flat row per trial.
    fields = ["rank", "trial", "changed"] + [f"{k}{s}" for k in EVALS for s in ("", "_lo", "_hi")] + [
        "env_steps", "env_steps_per_s", "games", "offers_per_game", "trades_per_game", "accept_rate", "lead_slope"]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields + list(plan["defaults"]))
        for rank, r in enumerate(rows, 1):
            t, s = r["trading"], r["training"]
            w.writerow([rank, r["trial"], changed(r, plan)]
                       + [x for k in EVALS for x in (r["eval"][k]["rate"], *r["eval"][k]["ci95"])]
                       + [s["env_steps"], round(s["env_steps_per_s"]), s["games"], t["offers_per_game"],
                          t["trades_per_game"], (t["accept_rate"] or [None])[0], (t["lead_slope"] or [None])[0]]
                       + [r["settings"].get(k) for k in plan["defaults"]])

    # report.md
    best = rows[0]
    m, c, v = plan["machine"], plan["code"], plan["versions"]
    reference = ("the hand-set defaults; the others are random draws from the search space (seed "
                 f"{plan['draw_seed']})" if plan["stage"] == 1 else
                 "the stage-1 winner at its own size; the others are the same settings at every other size")
    lines = [f"# Hyperparameter search: {plan['algo'].upper()}, stage {plan['stage']} (`{plan['search']}`)", "",
             f"Started {plan['created']}; {len(rows)} of {plan['trials']} trials finished. Code {c['commit'][:10]}"
             + (" with uncommitted changes (`code.diff`)" if c["dirty"] else "")
             + f"; fastCatan {v['fastcatan']}, torch {v['torch']}; {m['gpu']}, {m['cpus']} CPUs (`{m['host']}`).", "",
             "## Protocol", "",
             f"- {plan['trials']} trials of {plan['hours']} h of training each, one at a time, trades on, seed "
             f"{plan['trial_seed']} for every trial. League snapshots every {plan['defaults']['SNAPSHOT_EVERY']:,} "
             f"games, self-play share P_SELFPLAY = {plan['defaults']['P_SELFPLAY']}.",
             f"- Trial 00 is the reference: {reference}.",
             f"- Each final checkpoint plays {plan['eval_games']} games on the same boards (seed {plan['eval_seed']}), "
             "2v2 with the seats swapped, against " + "; ".join(plan["evals"].values())
             + ". Per-seat win rate with a Wilson 95% interval (chance 0.25; truncated games count for nobody).",
             f"- Trading: {plan['selfplay_games']} games of each agent against itself, with trade logs (metrics.py).",
             f"- Ranking by {NAMES[key]}, ties broken by the other two.", "",
             "## Search space", "", "| setting | default | searched |", "|---|---|---|"]
    for name, spec in plan["space"].items():
        searched = f"log-uniform {spec[1]:g} to {spec[2]:g}" if spec[0] == "log" else ", ".join(f"{x:g}" for x in spec)
        lines.append(f"| {name} | {plan['defaults'][name]:g} | {searched} |")
    lines += ["", "## Results", "",
              "| rank | trial | settings | vs reference | vs AB K=1 | vs random | env steps | steps/s | games | "
              "trades/game | acceptance |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rank, r in enumerate(rows, 1):
        t, s = r["trading"], r["training"]
        lines.append(f"| {rank} | {label(r)} | {changed(r, plan)} | " + " | ".join(interval(r["eval"][k]) for k in EVALS)
                     + f" | {s['env_steps'] / 1e6:.0f}M | {s['env_steps_per_s'] / 1e3:.0f}k | {s['games']:,} | "
                     f"{t['trades_per_game']} | {(t['accept_rate'] or ['-'])[0]} |")
    verdict = ("The reference itself ranks first." if best["trial"] == 0 else
               "Its interval lies above 0.25: it beats the reference." if best["eval"][key]["ci95"][0] > 0.25 else
               "Its interval includes 0.25: with these games it is not distinguishable from the reference.")
    lines += ["", f"**Winner: trial {best['trial']:02d}** ({changed(best, plan)}), {NAMES[key]} "
              f"{interval(best['eval'][key])}. {verdict}"]
    if ref and best is not ref:
        lines.append(f"Against Alpha-Beta K=1: winner {interval(best['eval']['vs_ab1'])}, reference "
                     f"{interval(ref['eval']['vs_ab1'])}.")
    lines += ["", "## Figures", ""] + [f"![{n}](figures/{n}.png)\n\n*{cap}* (`figures/{n}.pdf`)\n" for n, cap in written]
    lines += ["## Data", "",
              "- `plan.json`: the protocol, every trial's config, trainer defaults, versions, machine.",
              "- `trials.jsonl`: one row per trial (`eval.<opponent>.rate / ci95 / wins / games`, `trading.*`, "
              "`training.*`, `settings`).",
              "- `results.csv`: the results table, flat, with every setting.",
              f"- `runs/hp_{plan['algo']}_{plan['search']}_NN/metrics.jsonl`: each trial's learning curve; "
              "`selfplay.A*.fcev`: its self-play trade logs (`metrics.py` reads them).", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"report -> {out / 'report.md'} ({len(written)} figures)", flush=True)


if __name__ == "__main__":
    main()
