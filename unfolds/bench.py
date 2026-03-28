"""Multi-experiment benchmark runner with shared sanctified holdout.

Sits above :class:`Experiment` to:

1. Own the sanctified split so all experiments share the exact same holdout.
2. Hold multiple named experiments and compare them side-by-side.
3. Centralize CLI argument parsing (common flags + domain-specific).

Usage::

    from unfolds.bench import Research
    from unfolds.experiment import ExperimentConfig

    research = Research("Superconductor Tc", load_superconductor, TC_CONFIG)
    research.new_experiment("cascade", model_fn, after=report_routing)

    # CLI (terminal entry point)
    research.main()

    # Programmatic (inline / notebook)
    research.run()
    research.run(quick=True)
    research.experiment("cascade").run()
"""

import argparse
import json
import os
import pickle
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Optional

import numpy as np

_SNAPSHOT_WARN_BYTES = 5 * 1024 * 1024  # 5 MB

from .experiment import (
    Experiment, ExperimentConfig, _scores, _bracket_scores,
)


# ---------------------------------------------------------------------------
# RunContext — resolved config + CLI state passed to model factories
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """Passed to model factories -- carries resolved config + CLI state.

    Attributes
    ----------
    config : ExperimentConfig
        Resolved experiment config (after CLI overrides).
    quick : bool
        Whether ``--quick`` was passed.
    data_dir : str or None
        Data directory from ``--data-dir``.
    extra : dict
        Domain-specific CLI args (via ``research.add_argument``).
    feature_names : list or None
        Feature names from the dataset (populated by Research.run).
    """
    config: ExperimentConfig
    quick: bool = False
    data_dir: Optional[str] = None
    extra: dict = field(default_factory=dict)
    feature_names: Optional[list] = None


# ---------------------------------------------------------------------------
# Research — multi-experiment benchmark runner
# ---------------------------------------------------------------------------

class Research:
    """Multi-experiment benchmark runner with shared sanctified holdout.

    Owns one SanctifiedDataset -- all registered experiments run dev
    k-fold CV on the same dev pool and final evaluation on the same
    sanctified set.  The sanctified split is determined at dataset
    creation time and never changes.

    Three ways to run::

        # CLI (terminal)
        research.main()

        # Programmatic (inline / notebook)
        research.run()
        research.run(quick=True)

        # Single experiment
        research.experiment("cascade").run()

    Parameters
    ----------
    name : str
        Title (argparse description + output headers).
    loader : callable
        ``loader(data_dir, config)`` -> SanctifiedDataset.
    config : ExperimentConfig
        Base config (CLI flags may override fields).
    quick_k : int
        ``config.k`` when ``--quick`` is passed (default 3).
    """

    def __init__(self, name, loader, config, *, quick_k=3, save_dir=None,
                 notes_path=None):
        self._name = name
        self._loader = loader
        self._base_config = config
        self._quick_k = quick_k
        self._experiments = []      # list of (name, model_fn, after)
        self._extra_args = []       # list of (args, kwargs) for argparse
        self.history = []           # in-memory run records
        self._save_dir = save_dir   # disk persistence (runs.json)
        self._notes_path = notes_path  # research notes markdown

    def new_experiment(self, name, model_fn, *, after=None):
        """Register a named experiment.

        Parameters
        ----------
        name : str
            Experiment name (shown in output and comparison table).
        model_fn : callable
            ``model_fn(ctx: RunContext)`` -> model with ``fit(X, y)``
            / ``predict(X)``.  Called once per fold + once for final
            retrain.  Must return a fresh unfitted model each call.
        after : callable, optional
            ``after(model, final_data, ctx)`` -> None.
            Post-sanctified hook for domain-specific output
            (e.g. routing distribution, bias tables).

        Returns
        -------
        str
            UUID for this experiment definition.  Use with
            :meth:`rerun` to re-execute by ID.
        """
        exp_id = uuid.uuid4().hex[:8]
        self._experiments.append((name, model_fn, after, exp_id))
        return exp_id

    def experiment(self, name):
        """Return a Research scoped to a single experiment.

        The returned object shares loader, config, and extra args
        but only runs the named experiment::

            research.experiment("cascade").run(quick=True)

        Parameters
        ----------
        name : str
            Name of a registered experiment.

        Returns
        -------
        Research
            Scoped copy with only the named experiment.
        """
        matches = [(n, fn, af, eid) for n, fn, af, eid
                   in self._experiments if n == name]
        if not matches:
            names = [n for n, _, _, _ in self._experiments]
            raise KeyError(
                f"unknown experiment {name!r}, "
                f"available: {', '.join(names)}")
        scoped = Research(self._name, self._loader, self._base_config,
                          quick_k=self._quick_k, save_dir=self._save_dir,
                          notes_path=self._notes_path)
        scoped._experiments = matches
        scoped._extra_args = list(self._extra_args)
        scoped.history = self.history  # shared reference
        return scoped

    def rerun(self, exp_id, **kwargs):
        """Rerun an experiment by its definition UUID.

        Parameters
        ----------
        exp_id : str
            UUID returned by :meth:`new_experiment`.
        **kwargs
            Forwarded to :meth:`run` (``quick``, ``data_dir``, etc.).

        Returns
        -------
        dict
            ``{name: recap_dict}`` for the rerun experiment.
        """
        return self.experiment_by_id(exp_id).run(**kwargs)

    def experiment_by_id(self, exp_id):
        """Return a Research scoped to the experiment with the given UUID.

        Parameters
        ----------
        exp_id : str
            UUID returned by :meth:`new_experiment`.
        """
        matches = [(n, fn, af, eid) for n, fn, af, eid
                   in self._experiments if eid == exp_id]
        if not matches:
            available = {eid: n for n, _, _, eid in self._experiments}
            raise KeyError(
                f"unknown experiment id {exp_id!r}, "
                f"available: {available}")
        scoped = Research(self._name, self._loader, self._base_config,
                          quick_k=self._quick_k, save_dir=self._save_dir,
                          notes_path=self._notes_path)
        scoped._experiments = matches
        scoped._extra_args = list(self._extra_args)
        scoped.history = self.history
        return scoped

    @property
    def experiments(self):
        """List registered experiments as ``{name: id}`` dict."""
        return {n: eid for n, _, _, eid in self._experiments}

    def add_argument(self, *args, **kwargs):
        """Add domain-specific CLI arg (forwarded to argparse)."""
        self._extra_args.append((args, kwargs))

    # ---- programmatic entry point ----

    def run(self, *, quick=False, data_dir=None, folds=None, dedup=None):
        """Load data, run all registered experiments, return recaps.

        Programmatic alternative to :meth:`main` -- no CLI parsing::

            research.run()
            research.run(quick=True)
            research.experiment("cascade").run()

        Parameters
        ----------
        quick : bool
            Quick mode (fewer folds, forwarded to model factories).
        data_dir : str or None
            Data directory (forwarded to loader).
        folds : int or None
            Override number of folds.
        dedup : str or None
            Override dedup strategy (``'average'``, ``'keep-first'``,
            or ``None`` to disable).  Only applied when the base
            config has dedup set.

        Returns
        -------
        dict
            ``{name: recap_dict}`` for each experiment.
        """
        config = self._base_config
        overrides = {}
        if quick:
            overrides['k'] = self._quick_k
        if folds is not None:
            overrides['k'] = folds
        if dedup is not None and self._base_config.dedup is not None:
            overrides['dedup'] = dedup if dedup != 'none' else None
        if overrides:
            config = replace(config, **overrides)

        t0 = time.time()
        print(f"Loading {self._name}...")
        ds = self._loader(data_dir, config)
        load_time = time.time() - t0

        extra = getattr(self, '_cli_extra', {})
        ctx = RunContext(config=config, quick=quick, data_dir=data_dir,
                         extra=extra,
                         feature_names=list(ds.feature_names))
        results = self._run(ctx, ds, self._experiments)

        print(f"\n  Timing: load={load_time:.1f}s")
        return results

    # ---- CLI entry point ----

    def main(self, argv=None):
        """Parse CLI -> load data -> run all -> compare.

        Terminal entry point.  For programmatic use, call
        :meth:`run` instead.
        """
        # 1. Build parser
        parser = argparse.ArgumentParser(description=self._name)
        parser.add_argument("--data-dir", default=None,
                            help="Data directory")
        parser.add_argument("--quick", action="store_true",
                            help="Quick mode: fewer epochs, fewer folds")
        parser.add_argument("--folds", type=int, default=None,
                            help="Override number of folds")
        if self._base_config.dedup is not None:
            parser.add_argument(
                "--dedup",
                choices=["none", "average", "keep-first"],
                default=None,
                help="Duplicate handling override")
        parser.add_argument("--exp", default=None,
                            help="Run only this experiment (default: all)")

        for extra_args, extra_kwargs in self._extra_args:
            parser.add_argument(*extra_args, **extra_kwargs)

        args = parser.parse_args(argv)

        # 2. Scope to single experiment if requested
        target = self
        if args.exp is not None:
            matches = [n for n, _, _, _ in self._experiments if n == args.exp]
            if not matches:
                names = [n for n, _, _, _ in self._experiments]
                parser.error(
                    f"unknown experiment {args.exp!r}, "
                    f"available: {', '.join(names)}")
            target = self.experiment(args.exp)

        # 3. Delegate to run()
        dedup = getattr(args, 'dedup', None)
        extra = {k: v for k, v in vars(args).items()
                 if k not in ('data_dir', 'quick', 'folds', 'dedup', 'exp')}
        # Stash extra args so run() ctx picks them up
        target._cli_extra = extra

        return target.run(
            quick=args.quick,
            data_dir=args.data_dir,
            folds=args.folds,
            dedup=dedup,
        )

    # ---- core orchestration ----

    def _run(self, ctx, ds, experiments):
        """Run all experiments on the given dataset.

        Returns dict of {name: recap_dict}.
        """
        from .pipeline import Pipeline as _Pipeline

        config = ctx.config
        t_start = time.time()

        # Phase 1 — Dataset info header
        n_total = ds.n_samples
        n_dev = len(ds.X)
        n_sanct = ds.n_sanctified
        n_feat = ds.X.shape[1]
        print(f"  {n_total} samples, {n_feat} features")
        print(f"  Dev: {n_dev}, Sanctified: {n_sanct}")

        # Phase 2 — Dev evaluation for each experiment
        group_by = config.group_by
        dev_strategy = config.dev_strategy
        exp_objects = {}   # name -> Experiment
        for exp_name, model_fn, _, _ in experiments:
            t1 = time.time()

            exp = Experiment(ds, config=config)
            is_pipeline = None

            if dev_strategy == 'holdout':
                frac = config.holdout_fraction
                n_tr = int(n_dev * (1 - frac))
                n_te = n_dev - n_tr
                print(f"\n  [{exp_name}] Dev holdout "
                      f"(train={n_tr}, val={n_te})...")
                fold_iter = exp.holdout(fraction=frac)
            else:
                cv_label = "grouped " if group_by else ""
                print(f"\n  [{exp_name}] Dev {config.k}-fold "
                      f"{cv_label}CV...")
                fold_iter = exp.folds(group_by=group_by)

            for fold in fold_iter:
                model = model_fn(ctx)

                if is_pipeline is None:
                    is_pipeline = isinstance(model, _Pipeline)

                if is_pipeline:
                    model.fit(fold.X_train, fold.y_train,
                              exp.feature_names, X_val=fold.X_val)
                    pred = model.predict()
                else:
                    model.fit(fold.X_train, fold.y_train)
                    pred = model.predict(fold.X_val)

                fold.record(pred)
                mae = float(np.abs(pred - np.asarray(fold.y_val)).mean())
                if dev_strategy == 'holdout':
                    print(f"    Holdout MAE={mae:.2f}")
                else:
                    print(f"    Fold {fold.fold}: MAE={mae:.2f}")

            cv_time = time.time() - t1
            print(f"    Time: {cv_time:.1f}s")

            exp_objects[exp_name] = (exp, is_pipeline)

        # Phase 3 — One-shot sanctified evaluation (ALL experiments)
        final = ds.final_evaluate()
        dev_X = np.asarray(ds.X)
        dev_y = np.asarray(ds.y)

        final_models = {}
        for exp_name, model_fn, after_fn, _ in experiments:
            print(f"\n  [{exp_name}] Retraining on full dev set "
                  f"({len(dev_X)} samples)...")
            t2 = time.time()

            exp, is_pipeline = exp_objects[exp_name]

            model = model_fn(ctx)
            if is_pipeline:
                model.fit(dev_X, dev_y,
                          exp.feature_names, X_val=final.X_sanct)
                pred = np.asarray(model.predict(), dtype=np.float64)
            else:
                model.fit(dev_X, dev_y)
                pred = np.asarray(model.predict(final.X_sanct),
                                  dtype=np.float64)

            retrain_time = time.time() - t2
            print(f"    Retrain time: {retrain_time:.1f}s")

            # Inject sanctified metadata into Experiment
            exp._set_sanctified(final.sanct_idx, final.y_sanct)
            exp.record_final(pred)

            final_models[exp_name] = model

            # Post-sanctified hook
            if after_fn is not None:
                after_fn(model, final, ctx)

        # Phase 4 — Per-experiment recaps
        recaps = {}
        for exp_name, _, _, _ in experiments:
            exp, _ = exp_objects[exp_name]
            if len(experiments) > 1:
                print(f"\n  [{exp_name}]")
            recaps[exp_name] = exp.recap()

        # Phase 5 — Comparison table
        if len(experiments) > 1:
            _print_comparison(recaps)

        # Phase 6 — Auto-save snapshots
        if self._save_dir is not None:
            self._save_snapshots(
                exp_objects, final_models, experiments, dev_X, dev_y)

        # Phase 7 — Record stats
        elapsed = time.time() - t_start
        exp_ids = {n: eid for n, _, _, eid in experiments}
        record = _build_record(self._name, ctx, recaps, n_total, n_feat,
                               n_dev, n_sanct, elapsed, exp_ids)
        self.history.append(record)
        self._save_history()

        return recaps

    # ---- snapshot persistence ----

    def _save_snapshots(self, exp_objects, final_models, experiments,
                        dev_X, dev_y):
        """Auto-save fitted model + OOF predictions for each experiment."""
        snap_dir = os.path.join(self._save_dir, 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)

        for exp_name, _, _, _ in experiments:
            exp, _ = exp_objects[exp_name]
            model = final_models[exp_name]

            # Reconstruct OOF predictions from fold objects
            n_dev = len(dev_y)
            oof_pred = np.full(n_dev, np.nan)
            recorded = [f for f in exp._folds if f.recorded]
            for fold in recorded:
                oof_pred[fold.te_idx] = fold._predictions

            # Sanctified predictions
            sanct_pred = getattr(exp, '_final_predictions', None)

            # Pickle the model to check size
            model_bytes = pickle.dumps(model)
            size_mb = len(model_bytes) / (1024 * 1024)

            if size_mb > _SNAPSHOT_WARN_BYTES / (1024 * 1024):
                print(f"  [{exp_name}] Snapshot is {size_mb:.1f} MB "
                      f"(>{_SNAPSHOT_WARN_BYTES / (1024*1024):.0f} MB) "
                      f"— skipping auto-save. "
                      f"Use research.save_snapshot('{exp_name}', force=True) "
                      f"to save anyway.")
                continue

            # Save model pickle
            model_path = os.path.join(snap_dir, f'{exp_name}_model.pkl')
            with open(model_path, 'wb') as f:
                f.write(model_bytes)

            # Save arrays
            arrays_path = os.path.join(snap_dir, f'{exp_name}.npz')
            save_kw = dict(oof_pred=oof_pred, y_dev=dev_y)
            if sanct_pred is not None:
                save_kw['sanct_pred'] = sanct_pred
            np.savez_compressed(arrays_path, **save_kw)

            print(f"  [{exp_name}] Snapshot saved ({size_mb:.2f} MB) "
                  f"→ {snap_dir}/")

    def load_snapshot(self, exp_name):
        """Load a saved snapshot for an experiment.

        Returns
        -------
        dict
            ``{'model': fitted_model, 'oof_pred': array, 'y_dev': array,
               'sanct_pred': array_or_None}``.

        Raises
        ------
        FileNotFoundError
            If no snapshot exists for the given experiment.
        """
        if self._save_dir is None:
            raise ValueError("no save_dir configured")

        snap_dir = os.path.join(self._save_dir, 'snapshots')
        model_path = os.path.join(snap_dir, f'{exp_name}_model.pkl')
        arrays_path = os.path.join(snap_dir, f'{exp_name}.npz')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"no snapshot for '{exp_name}' at {model_path}")

        with open(model_path, 'rb') as f:
            try:
                model = pickle.load(f)
            except (AttributeError, ModuleNotFoundError) as e:
                import warnings
                warnings.warn(
                    f"Could not unpickle model for '{exp_name}': {e}. "
                    f"OOF/sanctified predictions are still available. "
                    f"Import the module defining the model class first.",
                    stacklevel=2)
                model = None

        data = np.load(arrays_path, allow_pickle=True)
        result = {
            'model': model,
            'oof_pred': data['oof_pred'],
            'y_dev': data['y_dev'],
        }
        if 'sanct_pred' in data:
            result['sanct_pred'] = data['sanct_pred']
        return result

    # ---- history persistence ----

    def _save_path(self):
        """Path to runs.json, or None if save_dir not set."""
        if self._save_dir is None:
            return None
        return os.path.join(self._save_dir, 'runs.json')

    def _save_history(self):
        """Append latest record to runs.json on disk."""
        path = self._save_path()
        if path is None:
            return
        # Load existing records from disk (may include runs from
        # previous sessions that aren't in self.history)
        existing = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(self.history[-1])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(existing, f, indent=2)

    def load_history(self):
        """Load run history from disk into ``self.history``.

        Merges on-disk records with any in-memory records (by
        timestamp dedup).  Returns ``self`` for chaining.
        """
        path = self._save_path()
        if path is None or not os.path.exists(path):
            return self
        try:
            with open(path) as f:
                disk = json.load(f)
        except (json.JSONDecodeError, OSError):
            return self
        # Merge: keep existing in-memory, add disk records not yet present
        mem_ts = {r['timestamp'] for r in self.history}
        for r in disk:
            if r.get('timestamp') not in mem_ts:
                self.history.append(r)
        self.history.sort(key=lambda r: r.get('timestamp', ''))
        return self

    def runs(self):
        """Query interface over run history.

        Returns a :class:`RunHistory` that can be filtered and
        queried::

            research.runs().table()
            research.runs().experiment("cascade").best("sanctified_mae")
            research.runs().full().latest()
            research.runs().compare("cascade", "bigger")

        Loads from disk first if ``save_dir`` is set.
        """
        self.load_history()
        return RunHistory(self.history)

    # ---- research notes ----

    def note(self, text, *, run=None):
        """Append a timestamped research note with auto-embedded stats.

        Writes to the markdown file at ``notes_path``.  If a run
        was just completed, its stats are embedded automatically.

        Parameters
        ----------
        text : str
            Your observation / interpretation.
        run : dict, optional
            Run record to embed.  Defaults to the latest entry in
            ``self.history`` (i.e. the run you just did).

        Example::

            research.run(quick=True)
            research.note("bigger router helps mid-bracket but not low")
        """
        if self._notes_path is None:
            raise ValueError(
                "no notes_path configured — pass notes_path= to Research()")

        if run is None and self.history:
            run = self.history[-1]

        entry = _format_note(text, run)

        # Append to file
        os.makedirs(os.path.dirname(self._notes_path) or '.', exist_ok=True)
        header_needed = not os.path.exists(self._notes_path)
        with open(self._notes_path, 'a') as f:
            if header_needed:
                f.write(f"# {self._name} — Research Notes\n\n")
            f.write(entry)

    def notes(self):
        """Read and print the research notes file."""
        if self._notes_path is None:
            print("  No notes_path configured.")
            return None
        if not os.path.exists(self._notes_path):
            print(f"  No notes yet ({self._notes_path}).")
            return None
        with open(self._notes_path) as f:
            content = f.read()
        print(content)
        return content


# ---------------------------------------------------------------------------
# Note formatter
# ---------------------------------------------------------------------------

def _format_note(text, run):
    """Format a research note entry with embedded run stats."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"### {ts} — {text}\n\n"]

    if run is not None:
        exps = run.get('experiments', {})
        if exps:
            lines.append("| experiment | dev MAE | sanctified MAE | R\u00b2 |\n")
            lines.append("|---|---|---|---|\n")
            for name, stats in exps.items():
                if stats is None:
                    lines.append(f"| {name} | - | - | - |\n")
                    continue
                dev = f"{stats['dev_mae']:.2f} \u00b1{stats['dev_mae_std']:.2f}"
                sanct = (f"{stats['sanctified_mae']:.2f}"
                         if 'sanctified_mae' in stats else "-")
                r2 = (f"{stats.get('sanctified_r2', stats.get('dev_r2', 0)):.4f}")
                lines.append(f"| {name} | {dev} | {sanct} | {r2} |\n")
            lines.append("\n")

        config = run.get('config', {})
        parts = []
        parts.append(f"k={config.get('k', '?')}")
        parts.append(f"seed={config.get('seed', '?')}")
        if config.get('dedup'):
            parts.append(f"dedup={config['dedup']}")
        if run.get('quick'):
            parts.append("quick=True")
        parts.append(f"n={run.get('n_samples', '?')}")
        lines.append(f"Config: {', '.join(parts)}\n\n")
    else:
        lines.append("\n")

    lines.append("---\n\n")
    return ''.join(lines)


# ---------------------------------------------------------------------------
# RunHistory — queryable run history
# ---------------------------------------------------------------------------

class RunHistory:
    """Queryable, chainable view over run records.

    Filters return new ``RunHistory`` objects (immutable chain).
    Terminal methods (``.best()``, ``.latest()``, ``.table()``,
    ``.compare()``) produce output.

    Obtain via ``research.runs()``.
    """

    def __init__(self, records):
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def __repr__(self):
        return f"RunHistory({len(self._records)} runs)"

    # ---- filters (return new RunHistory) ----

    def experiment(self, name):
        """Keep only runs that contain the named experiment."""
        return RunHistory([
            r for r in self._records
            if name in r.get('experiments', {})])

    def full(self):
        """Keep only non-quick runs."""
        return RunHistory([
            r for r in self._records if not r.get('quick', False)])

    def quick(self):
        """Keep only quick runs."""
        return RunHistory([
            r for r in self._records if r.get('quick', False)])

    def last(self, n=1):
        """Keep the last *n* runs (by timestamp)."""
        sorted_recs = sorted(
            self._records, key=lambda r: r.get('timestamp', ''))
        return RunHistory(sorted_recs[-n:])

    def since(self, timestamp):
        """Keep runs after *timestamp* (ISO string or datetime)."""
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        return RunHistory([
            r for r in self._records
            if r.get('timestamp', '') >= timestamp])

    # ---- terminals ----

    def latest(self):
        """Return the most recent run record (single dict)."""
        if not self._records:
            return None
        return max(self._records,
                   key=lambda r: r.get('timestamp', ''))

    def best(self, metric='sanctified_mae', experiment=None, *,
             higher_is_better=None):
        """Return the run record with the best metric value.

        Parameters
        ----------
        metric : str
            Key inside the experiment stats dict.  Common values:
            ``'dev_mae'``, ``'sanctified_mae'``, ``'dev_r2'``,
            ``'sanctified_r2'``.
        experiment : str, optional
            Which experiment's metric to use.  If None, uses the
            first experiment in each run.
        higher_is_better : bool, optional
            If None, auto-detected: True for ``r2``, False for
            ``mae``/``rmse``.
        """
        if higher_is_better is None:
            higher_is_better = 'r2' in metric

        best_rec = None
        best_val = None
        for r in self._records:
            exps = r.get('experiments', {})
            if experiment is not None:
                stats = exps.get(experiment)
            else:
                stats = next(iter(exps.values()), None) if exps else None
            if stats is None:
                continue
            val = stats.get(metric)
            if val is None:
                continue
            if best_val is None:
                best_rec, best_val = r, val
            elif higher_is_better and val > best_val:
                best_rec, best_val = r, val
            elif not higher_is_better and val < best_val:
                best_rec, best_val = r, val
        return best_rec

    def compare(self, *exp_names):
        """Print cross-experiment comparison across all runs.

        Shows each experiment's best sanctified MAE and latest MAE,
        plus the trend (delta from previous run).

        Parameters
        ----------
        *exp_names : str
            Experiment names to compare.  If empty, uses all
            experiments found across all runs.
        """
        if not self._records:
            print("  No runs in history.")
            return

        # Discover all experiment names if none given
        if not exp_names:
            exp_names = []
            seen = set()
            for r in self._records:
                for name in r.get('experiments', {}):
                    if name not in seen:
                        exp_names.append(name)
                        seen.add(name)

        W = 68
        print(f"\n{'=' * W}")
        print(f"  CROSS-EXPERIMENT ANALYSIS  ({len(self._records)} runs)")
        print(f"  {'-' * (W - 4)}")
        r2_hdr = "Best R\u00b2"
        print(f"  {'':>16s}  {'Best MAE':>10s}  {'Latest MAE':>10s}  "
              f"{'Delta':>8s}  {r2_hdr:>8s}")
        print(f"  {'-' * (W - 4)}")

        for name in exp_names:
            exp_runs = [r for r in self._records
                        if name in r.get('experiments', {})]
            if not exp_runs:
                print(f"  {name:>16s}  {'(no data)':>10s}")
                continue

            # Sort by timestamp
            exp_runs.sort(key=lambda r: r.get('timestamp', ''))

            # Extract sanctified MAEs (fall back to dev)
            def _mae(r):
                s = r['experiments'][name]
                if s is None:
                    return None
                return s.get('sanctified_mae', s.get('dev_mae'))

            def _r2(r):
                s = r['experiments'][name]
                if s is None:
                    return None
                return s.get('sanctified_r2', s.get('dev_r2'))

            maes = [_mae(r) for r in exp_runs]
            maes_valid = [m for m in maes if m is not None]

            if not maes_valid:
                print(f"  {name:>16s}  {'(no data)':>10s}")
                continue

            best_mae = min(maes_valid)
            latest_mae = maes[-1] if maes[-1] is not None else maes_valid[-1]
            best_r2_val = max(
                (r2 for r2 in (_r2(r) for r in exp_runs) if r2 is not None),
                default=None)

            # Delta from previous run
            delta_str = ""
            if len(maes_valid) >= 2:
                prev = maes[-2] if maes[-2] is not None else maes_valid[-2]
                delta = latest_mae - prev
                arrow = "+" if delta > 0 else ""
                delta_str = f"{arrow}{delta:.2f}"

            r2_str = f"{best_r2_val:.4f}" if best_r2_val is not None else "-"
            print(f"  {name:>16s}  {best_mae:>10.2f}  {latest_mae:>10.2f}  "
                  f"{delta_str:>8s}  {r2_str:>8s}")

        print(f"  {'-' * (W - 4)}")

    def table(self):
        """Print a timeline table of all runs.

        One row per run, showing timestamp, quick flag, and
        per-experiment MAE.
        """
        if not self._records:
            print("  No runs in history.")
            return

        # Discover all experiment names across all runs
        all_exps = []
        seen = set()
        for r in self._records:
            for name in r.get('experiments', {}):
                if name not in seen:
                    all_exps.append(name)
                    seen.add(name)

        sorted_recs = sorted(
            self._records, key=lambda r: r.get('timestamp', ''))

        # Header
        exp_cols = "  ".join(f"{n:>12s}" for n in all_exps)
        print(f"\n  {'#':>3s}  {'Timestamp':>19s}  {'Mode':>5s}  {exp_cols}")
        sep_w = 3 + 2 + 19 + 2 + 5 + len(all_exps) * 14
        print(f"  {'-' * sep_w}")

        for i, r in enumerate(sorted_recs):
            ts = r.get('timestamp', '?')[:19]  # trim tz
            mode = "quick" if r.get('quick') else "full"

            vals = []
            for name in all_exps:
                stats = r.get('experiments', {}).get(name)
                if stats is None:
                    vals.append(f"{'  -':>12s}")
                else:
                    mae = stats.get('sanctified_mae', stats.get('dev_mae'))
                    if mae is not None:
                        vals.append(f"{mae:>12.2f}")
                    else:
                        vals.append(f"{'  -':>12s}")

            val_str = "  ".join(vals)
            print(f"  {i:>3d}  {ts:>19s}  {mode:>5s}  {val_str}")


# ---------------------------------------------------------------------------
# Run record builder
# ---------------------------------------------------------------------------

def _build_record(name, ctx, recaps, n_total, n_feat, n_dev, n_sanct,
                  elapsed, exp_ids=None):
    """Build a JSON-serializable stats record for one run."""
    config = ctx.config

    exp_stats = {}
    for exp_name, recap in recaps.items():
        if recap is None:
            exp_stats[exp_name] = None
            continue
        dev = recap['dev']
        entry = {
            'dev_mae': dev['mae']['mean'],
            'dev_mae_std': dev['mae']['std'],
            'dev_r2': dev['r2']['mean'],
            'dev_r2_std': dev['r2']['std'],
            'dev_rmse': dev['rmse']['mean'],
        }
        sanct = recap.get('sanctified')
        if sanct is not None:
            entry['sanctified_mae'] = sanct['mae']
            entry['sanctified_r2'] = sanct['r2']
            entry['sanctified_rmse'] = sanct['rmse']
        if exp_ids and exp_name in exp_ids:
            entry['id'] = exp_ids[exp_name]
        exp_stats[exp_name] = entry

    # Convert brackets (tuples) to lists for JSON
    brackets = None
    if config.brackets is not None:
        brackets = [list(b) for b in config.brackets]

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'name': name,
        'config': {
            'seed': config.seed,
            'k': config.k,
            'sanctified_fraction': config.sanctified_fraction,
            'dedup': config.dedup,
            'brackets': brackets,
        },
        'quick': ctx.quick,
        'n_samples': n_total,
        'n_features': n_feat,
        'n_dev': n_dev,
        'n_sanctified': n_sanct,
        'experiments': exp_stats,
        'elapsed_s': round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Comparison table formatter
# ---------------------------------------------------------------------------

def _print_comparison(recaps):
    """Print side-by-side comparison of multiple experiments."""
    W = 54
    print(f"\n{'=' * W}")
    print(f"  COMPARISON")
    print(f"  {'-' * (W - 4)}")

    # Header
    print(f"  {'':>16s}  {'Dev MAE (+/-std)':>18s}  {'Sanctified MAE':>14s}")
    print(f"  {'-' * (W - 4)}")

    for name, recap in recaps.items():
        if recap is None:
            print(f"  {name:>16s}  {'(no data)':>18s}")
            continue

        dev = recap['dev']
        dev_mae = dev['mae']['mean']
        dev_std = dev['mae']['std']
        dev_str = f"{dev_mae:.2f} +/-{dev_std:.2f}"

        sanct = recap.get('sanctified')
        if sanct is not None:
            sanct_str = f"{sanct['mae']:.2f}"
        else:
            sanct_str = "-"

        print(f"  {name:>16s}  {dev_str:>18s}  {sanct_str:>14s}")

    print(f"  {'-' * (W - 4)}")
