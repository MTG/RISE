"""The ``rise`` command line.

One sub-command per experiment, assembled from the experiment modules themselves.
Defaults come from ``configs.yaml`` so that a run is reproducible from the file
alone, and any flag given on the command line overrides it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rich.traceback import install as install_rich_traceback

from .config import load_config
from .console import console
from .experiments import EXPERIMENTS
from .figures.style import apply_style
from .reproducibility import DEFAULT_SEED, resolve_device, seed_everything

PROGRAM = "rise"

EPILOG = """\
Experiments are listed in the order they have to be run: `preprocess` builds every
dataset, `pretrain` produces the encoder the four downstream experiments load.
"""


def build_parser(config: dict[str, dict[str, object]] | None = None) -> argparse.ArgumentParser:
    """Assemble the parser, seeding each sub-command's defaults from the config."""
    config = load_config() if config is None else config

    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="RISE — a rāga-independent encoder for svara representation in Carnatic music",
        epilog=EPILOG,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for every source of randomness")
    parser.add_argument("--device", default=None, help="torch device; defaults to CUDA when available")

    subparsers = parser.add_subparsers(dest="experiment", required=True, metavar="experiment")
    for name, module in EXPERIMENTS.items():
        subparser = subparsers.add_parser(
            name,
            help=module.DESCRIPTION,
            description=module.__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(**_checked_defaults(subparser, name, config.get(name, {})))
        subparser.set_defaults(_run=module.run)

    return parser


def _checked_defaults(
    parser: argparse.ArgumentParser,
    experiment: str,
    values: dict[str, object],
) -> dict[str, object]:
    """Drop configuration keys that no option accepts, and say so.

    ``set_defaults`` happily invents an attribute for an unknown key, so a typo in
    ``configs.yaml`` would otherwise be silently ignored at the point where it
    matters and reported as a parameter of the run.
    """
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(values) - known)
    for key in unknown:
        console.print(f"[warning]![/warning] configs.yaml: {experiment}.{key} is not an option; ignoring it")
    return {key: value for key, value in values.items() if key in known}


def main(argv: Sequence[str] | None = None) -> int:
    install_rich_traceback(console=console, show_locals=False, suppress=["torch"])
    args = build_parser().parse_args(argv)

    seed_everything(args.seed)
    apply_style()

    args.device = str(resolve_device(args.device))

    try:
        args._run(args)
    except FileNotFoundError as error:
        console.print(f"[error]Missing input:[/error] {error.filename}")
        console.print("[detail]Run `./run.sh preprocess` first, and `./run.sh pretrain` for the encoder.[/detail]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted.[/warning] Training experiments resume from their last epoch.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
