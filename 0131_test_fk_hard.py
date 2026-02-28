import argparse

import train_multi as tm


def build_parser() -> argparse.ArgumentParser:
    parser = tm.build_arg_parser()
    parser.description = (
        "Step2: Joint training with hard FK propagation only (FK not masked, "
        "no soft propagation, no fk loss)."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Force Step2 settings.
    args.mask_fk = 0
    args.use_fk_propagation = 1
    args.use_soft_fk_propagation = 0
    args.weight_fk = 0.0
    args.use_known_mask = 1

    tm.train(args)


if __name__ == "__main__":
    main()
