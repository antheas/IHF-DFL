from datetime import datetime
from pathlib import Path
from ml.visualization import visualize_simulation
import json
import os
import multiprocessing as mp
import re
import sys
import traceback
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Running this file directly puts experiment/ on sys.path.
# Insert src/ and repo root before any project imports so they resolve
# regardless of whether the editable install is present.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

from utils.paths import repo_root
REPO_ROOT = repo_root(Path(__file__))

import experiment.experiment_runner as ExperimentRunner
from itertools import product
from dataclasses import dataclass
from experiment.helper import getPath, create_run_ids, resolve_attack_params
from experiment.experiment_configuration import ExperimentConfiguration
from experiment.experiment_presets import PRESETS
from utils.async_writer import AsyncWriter
from selector import choose_from_list
from analysis import ExperimentLogger

DATA_ROOT = REPO_ROOT / "data"
DATASETSLOW = "cifar.10"
DATASETFAST = "mnist"
RESULTDATAFOLDER = REPO_ROOT / "data" / "runs" / "experiments"

# ---------------- PRESET SEARCH SPACE ----------------

preset = "test"
_use_defaults = True
datasets = [ DATASETFAST ]


# ---------------- OUTPUT ----------------

OUTPUTHEADERS = [
    "round",
    "time",
    "globalAcc",
    "globalLoss",
    "GRS",
    "accAvgPerUser",
    "lossAvgPerUser",
    "rewards",
    "conctractBalanceRewards",
    "punishments",
    "contributionScores",
    "feedbackMatrix",
    "disqualifiedUsers",
    "userStatuses",
    "GasTransactions",
]

WRITERBUFFERSIZE = 200


# ---------------- SKIP STRUCTURE ----------------

@dataclass(frozen=True)
class Skip:
    preset: str
    dataset: str
    strategy: str
    outlier_detection: bool
    freerider_activation_round: int | None
    freerider_noise: float | None
    malicious_activation_round: int | None
    malicious_noise: float | None
    run_id : int

skips: list[Skip] = []


# ---------------- MAIN ----------------

def build_run_configs(preset_name, datasets) -> list[tuple[ExperimentConfiguration, int]]:
    """
    Expand a preset into its per-run configurations, in driver order.
    Returns (config, run_id) pairs; config holds every value used for the run name and metadata.
    """
    preset_config = PRESETS[preset_name]
    productVar = []

    malicious_rounds = (
        preset_config.malicious_start_round
        if preset_config.malicious_start_round is not None
        else [None]
    )

    malicious_noises = (
        preset_config.malicious_noise_scale
        if preset_config.malicious_noise_scale is not None
        else [None]
    )


    freerider_rounds = (
        preset_config.freerider_start_round
        if preset_config.freerider_start_round is not None
        else [None]
    )

    freerider_noises = (
        preset_config.freerider_noise_scale
        if preset_config.freerider_noise_scale is not None
        else [None]
    )


    number_of_runs = preset_config.number_of_runs if preset_config.number_of_runs is not None else 1
    runs = create_run_ids(number_of_runs)
    if not runs:
        runs = [1]

    for (
            strategy,
            outlier_detection,
            freerider_round,
            freerider_noise,
            malicious_activation_round,
            malicious_noise,
            dataset,
            run_id
    ) in product(
        preset_config.contribution_score_strategy,
        preset_config.use_outlier_detection,
        freerider_rounds,
        freerider_noises,
        malicious_rounds,
        malicious_noises,
        datasets,
        runs
    ):

        productVar.append((
            strategy,
            outlier_detection,
            freerider_round,
            freerider_noise,
            # freerider_attack_type,
            malicious_activation_round,
            malicious_noise,
            # malicious_attack_type,
            dataset,
            # aggregation_rule,
            # data_distribution,
            run_id
        ))

    run_configs = []
    for (
        strategy,
        outlier_detection,
        freerider_round,
        freerider_noise,
        malicious_activation_round,
        malicious_noise,
        dataset,
        run_id
    ) in productVar:
        # None in a preset falls back to this run's free-rider values (#83), so no run is ever
        # built with an unresolved malicious participant.
        (
            freerider_round,
            freerider_noise,
            malicious_activation_round,
            malicious_noise,
        ) = resolve_attack_params(
            has_bad=preset_config.number_of_bad_contributors > 0,
            has_freerider=preset_config.number_of_freerider_contributors > 0,
            freerider_round=freerider_round,
            freerider_noise=freerider_noise,
            malicious_activation_round=malicious_activation_round,
            malicious_noise=malicious_noise,
            warn=False,
        )

        config = ExperimentConfiguration(preset=preset_name, use_defaults=_use_defaults)
        config.preset_name = preset_name
        config.dataset = dataset

        # override parameters for this run
        config.contribution_score_strategy = strategy
        config.use_outlier_detection = outlier_detection
        config.freerider_start_round = freerider_round
        config.freerider_noise_scale = freerider_noise
        config.malicious_start_round = malicious_activation_round
        config.malicious_noise_scale = malicious_noise
        run_configs.append((config, run_id))

    return run_configs


def skip_for(preset_name, config, run_id) -> "Skip":
    """The Skip entry --skipFolder matching uses for one run."""
    return Skip(
        preset=preset_name,
        dataset=config.dataset,
        strategy=config.contribution_score_strategy,
        outlier_detection=config.use_outlier_detection,
        freerider_activation_round=config.freerider_start_round,
        freerider_noise=config.freerider_noise_scale,
        malicious_activation_round=config.malicious_start_round,
        malicious_noise=config.malicious_noise_scale,
        run_id=run_id,
    )


def run_one(config, run_id, path, author, startTime):
    """Run one config and write its .csv/.pkl (or an error-*.txt) to path's folder."""
    dataset = config.dataset
    # Normalize before the writer thread logs the metadata; run_experiment does it too, but races the writer.
    config.dataset = dataset.replace(".", "-")
    try:
        writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, author)
        metadata = {**vars(config), "dataset": dataset, "timestamp": startTime}
        logger = ExperimentLogger(experiment_id=path.stem, metadata=metadata)
        ExperimentRunner.run_experiment(dataset, config, run_id, writer, logger)
        writer.finish()

        logger.save(path.with_suffix(".pkl"))
        return None
    except Exception as e:

        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        err_file = path.parent / f"error-{ts}.txt"

        config_as_dict = config.__dict__

        text = "CONFIG:\n"
        text += json.dumps(config_as_dict, indent=4, default=str)
        text += "\n\nEXCEPTION:\n"
        text += "".join(traceback.format_exception(e))

        err_file.parent.mkdir(parents=True, exist_ok=True)
        with open(err_file, "w") as f:
            f.write(text)

        print(f"Error logged to: {err_file}")
        return err_file


def run_one_logged(config, run_id, path, author, startTime, rpc_urls):
    """Parallel worker: same as run_one, with this run's output sent to <path>.log instead of the shared terminal.
    Borrows a node from rpc_urls for the duration of the run, so no two concurrent runs share a chain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rpc_url = rpc_urls.get()
    try:
        # require_env_var's load_dotenv does not override variables that are already set
        os.environ["RPC_URL"] = rpc_url
        with open(path.with_suffix(".log"), "w", buffering=1) as log:
            sys.stdout = sys.stderr = log
            print(f"Using node {rpc_url}")
            return run_one(config, run_id, path, author, startTime)
    finally:
        rpc_urls.put(rpc_url)


def main(author, parallel=1): # single preset
    startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    if args.skipFolder is not None:
        parseSkips()

    # A fixed output folder makes reruns resume: runs that already finished there are skipped.
    if args.outputName is not None:
        time = args.outputName
        addSkipsFromDir(RESULTDATAFOLDER / time)
    else:
        time = datetime.now().strftime("%d-%m-%y--%H_%M_%S")

    run_configs = build_run_configs(preset, datasets)
    total = len(run_configs)

    skipConfigs = [skip_for(preset, config, run_id) for config, run_id in run_configs]
    skipsCount = sum(1 for sc in skipConfigs if shouldSkip(sc))

    todo = []
    for (config, run_id), skipConfig in zip(run_configs, skipConfigs):
        if shouldSkip(skipConfig):
            print(f"Skipping: {skipConfig}")
            continue
        else:
            # debug
            print(f"NOT skipped: {skipConfig}")
            for s in skips:
                if s.run_id == skipConfig.run_id and s.strategy == skipConfig.strategy:
                    print(f"  candidate: {s}")

        path = getPath(config, time, config.dataset, preset, RESULTDATAFOLDER, run_id=run_id)
        todo.append((config, run_id, path))

    if parallel > 1:
        run_parallel(todo, parallel, author, startTime, skipsCount, total)
        return

    for done, (config, run_id, path) in enumerate(todo):
        progress_bar(done, skipsCount, total)
        run_one(config, run_id, path, author, startTime)

        # visualize_simulation(experiment.model, DATA_ROOT / "figures")

    #ExperimentRunner.print_transactions(experiment)


def run_parallel(todo, parallel, author, startTime, skipsCount, total):
    # Real-net runs sign with shared private keys and fetch nonces themselves, so concurrent runs would collide.
    if any(not config.fork for config, _, _ in todo):
        raise SystemExit("--parallel only supports fork=True (local Ganache) runs.")

    # Runs on one node share its accounts (nonce collisions) and its clock (evm_increaseTime), so each
    # concurrent run needs a node of its own.
    urls = [u.strip() for u in os.environ.get("RPC_URLS", "").split(",") if u.strip()]
    if len(urls) < parallel:
        raise SystemExit(f"--parallel {parallel} needs {parallel} separate Ganache nodes: "
                         f"set RPC_URLS=url1,url2,... (run_all.sh does this), got {len(urls)}.")

    print(f"Running {len(todo)} experiments, {parallel} at a time on {', '.join(urls[:parallel])}. "
          f"Per-run output goes to <result>.log")
    done = 0
    progress_bar(done, skipsCount, total)
    # spawn: no inherited torch/CUDA state; one task per child so every run starts from a clean process.
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager, \
            ProcessPoolExecutor(max_workers=parallel, mp_context=ctx, max_tasks_per_child=1) as pool:
        rpc_urls = manager.Queue()
        for url in urls[:parallel]:
            rpc_urls.put(url)
        futures = {pool.submit(run_one_logged, config, run_id, path, author, startTime, rpc_urls): path
                   for config, run_id, path in todo}
        try:
            for future in as_completed(futures):
                path = futures[future]
                done += 1
                try:
                    err_file = future.result()
                except Exception as e:  # worker died (e.g. OOM) before it could log the error itself
                    err_file = None
                    print(f"FAILED (worker crashed): {path.name}: {e!r}")
                if err_file is not None:
                    print(f"FAILED: {path.name} -> {err_file}")
                else:
                    print(f"Finished: {path.name}")
                progress_bar(done, skipsCount, total)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


# ---------------- ARGUMENTS ----------------

parser = argparse.ArgumentParser()
parser.add_argument("--skipFolder", type=str)
parser.add_argument("--author", type=str)
parser.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="number of experiments to run at the same time (default: 1)")
parser.add_argument("--preset", type=str, choices=[name for name in PRESETS if name != "default"],
                    help=f"preset to run (default: {preset})")
parser.add_argument("--dataset", type=str, action="append", choices=[DATASETFAST, DATASETSLOW],
                    help="dataset to run, repeatable (default: " + ", ".join(datasets) + ")")
parser.add_argument("--outputName", type=str,
                    help="fixed result subfolder instead of a timestamp; runs already finished in it are skipped")
# Lenient on import (e.g. pytest's argv); the script entry point below re-parses strictly.
args, _ = parser.parse_known_args()


# ---------------- SKIP PARSING ----------------

def parseSkips():
    if not RESULTDATAFOLDER.exists():
        return

    dirs = sorted([d for d in RESULTDATAFOLDER.iterdir() if d.is_dir()])

    chosenDirs = choose_from_list(
        dirs,
        "Select directories to scan for configs to skip",
        False
    )

    for dir in chosenDirs:
        addSkipsFromDir(dir)


def addSkipsFromDir(dir: Path):
    """Add a skip for every run that finished in dir (the .pkl is only saved on success)."""
    if not dir.exists():
        return

    for file in [p.name for p in dir.iterdir() if p.is_file() and p.suffix == ".pkl"]:
        skip = skip_from_filename(file)
        if skip is None:
            print("Did not match:", file)
            continue
        skips.append(skip)


def skip_from_filename(file: str) -> "Skip | None":
    """Parse a result file name into the Skip entry of the run that produced it."""
    m = re.fullmatch(
        r"(?P<preset>[^-]+)-"
        r"(?P<dataset>[^-]+)-"
        r"(?P<strategy>[^-]+)-"
        r"(?P<outlierDetection>[^-]+)-"
        r"(?P<freeriderRound>[^-]+)-"
        r"(?P<freeriderNoise>[^-]+)-"
        r"(?P<maliciousRound>[^-]+)-"
        r"(?P<maliciousNoise>[^-]+)-"
        r"-?(?P<runId>[0-9]+)?"  # <-- optional run ID part
        r"(?:-\{[0-9a-fA-F-]+\})?"  # <-- optional UUID part
        r"\.pkl",
        file,
    )
    if not m:
        return None

    mal_round = m.group("maliciousRound")
    mal_noise = m.group("maliciousNoise")

    return Skip(
        preset=m.group("preset"),
        dataset=m.group("dataset"),
        strategy=m.group("strategy"),
        outlier_detection=m.group("outlierDetection") == "True",
        freerider_activation_round=int(m.group("freeriderRound")),
        freerider_noise=float(m.group("freeriderNoise")),
        malicious_activation_round=int(mal_round) if mal_round != "None" else None,
        malicious_noise=float(mal_noise) if mal_noise != "None" else None,
        run_id=int(m.group("runId")) if m.group("runId") is not None else 0
    )


# ---------------- SKIP CHECK ----------------

def shouldSkip(config: Skip):
    # Exact match only. Run configs are always resolved (build_run_configs), so result files
    # with "None" malicious values come from the pre-#83 runs where the attack never happened
    # and must never count as a finished run.
    return config in skips


# ---------------- PROGRESS ----------------

def progress_bar(i, skipsCount, total):

    percent = (i / total) * 100
    totalPercent = ((i + skipsCount) / total) * 100

    print(
        f"Progress: {i}/{total} ({percent:.2f}%)\n"
        f"Total Progress: {i + skipsCount}/{total} ({totalPercent:.2f}%)\n",
        flush=True,
    )



# ---------------- ENTRY ----------------

if __name__ == "__main__":
    args = parser.parse_args()  # strict: fail on typos / unknown flags
    author = args.author if args.author is not None else input("Author?\n")
    if args.preset is not None:
        preset = args.preset
    if args.dataset is not None:
        datasets = args.dataset
    mp.freeze_support()
    if args.parallel < 1:
        parser.error("--parallel must be >= 1")
    main(author, args.parallel)
    print("Done :)")


