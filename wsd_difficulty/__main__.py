"""Command line: python -m wsd_difficulty <stage> [options]"""

import argparse

from . import config, pipeline
from .models import MODEL_NAMES

STAGES = ["parse-rst", "prepare", "run", "run-discourse", "analyse", "all"]


def main():
    parser = argparse.ArgumentParser(prog="wsd_difficulty",
                                     description="Sense entropy, context, discourse and WSD difficulty.")
    parser.add_argument("stage", choices=STAGES, help="which step to run ('all' runs prepare to analyse)")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes for run and run-discourse")
    parser.add_argument("--max-target-words", type=int, default=None, help="prepare: random sample of words, for quick runs")
    parser.add_argument("--device", default=None, help="parse-rst: mps, cuda or cpu (default: best available)")
    parser.add_argument("--limit", type=int, default=None, help="parse-rst: parse at most this many more documents")
    arguments = parser.parse_args()
    stage = arguments.stage

    if stage == "parse-rst":
        pipeline.parse_rst(config.FRAMEWORK_DIR, arguments.device, arguments.limit)
    if stage in ("prepare", "all"):
        pipeline.prepare(config.FRAMEWORK_DIR, arguments.max_target_words)
    if stage in ("run", "all"):
        pipeline.run(MODEL_NAMES, list(config.FEATURE_SETS), arguments.workers)
    if stage in ("run-discourse", "all"):
        pipeline.run_discourse(arguments.workers)
    if stage in ("analyse", "all"):
        pipeline.analyse()


if __name__ == "__main__":
    main()
