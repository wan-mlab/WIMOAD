from runner import parse_args


def test_parse_args_defaults_match_legacy_runner_behavior():
    args = parse_args([])

    assert args.group == "ca"
    assert args.omics == "both"
    assert args.cv == "LOO"
    assert args.output_dir == "."
    assert args.n_splits == 5
    assert args.n_repeats is None


def test_parse_args_accepts_configurable_workflow_options():
    args = parse_args(
        [
            "--group",
            "all",
            "--omics",
            "expression",
            "--cv",
            "KFold",
            "--output-dir",
            "results",
            "--n-splits",
            "3",
            "--n-repeats",
            "2",
        ]
    )

    assert args.group == "all"
    assert args.omics == "expression"
    assert args.cv == "KFold"
    assert args.output_dir == "results"
    assert args.n_splits == 3
    assert args.n_repeats == 2
