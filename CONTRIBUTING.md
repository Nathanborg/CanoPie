# Contributing to CanoPie

Thank you for helping improve CanoPie. Contributions of every size are welcome:
bug reports, documentation, test data, and code.

## Reporting a problem or asking for help

- Open an issue at <https://github.com/Nathanborg/CanoPie/issues>.
- Describe what you did, what you expected, and what happened. Include the
  CanoPie version (commit), your operating system, and, if you can, a small
  image or project that reproduces the problem.
- For questions about using CanoPie, open an issue with the `question` label.

## Proposing a change

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep it focused: one fix or feature per pull request.
3. Add or update tests in `canopie/qc/` for what you changed.
4. Run the test suite (below) and make sure it passes.
5. Open a pull request describing the problem and how you verified the fix.

## Running the tests

```sh
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest canopie/qc -q -p no:cacheprovider
```

A feature-level report is written to `canopie/qc/last_run_report.txt`. Run the
suite on a normal desktop session (not with `QT_QPA_PLATFORM=offscreen`): a few
viewer tests need a real window system. The full suite is large; on a laptop it
is faster to run it in parts, e.g. `canopie/qc/test_[a-o]*.py`, then
`canopie/qc/test_[p-s]*.py`, then the remaining files.

## Code style

Follow the style of the surrounding code. Image edits are never written into the
image files: they live in the `.ax` sidecar of each image and are replayed by
every consumer (viewer, exports, inspector). Keep new code consistent with that
rule, and keep modules without a GUI (e.g. `canopie/canopy_finder.py`) free of Qt
so they stay testable.

## Code of conduct

Be respectful and constructive. Harassment or discrimination of any kind is not
tolerated in issues, pull requests, or any other project space.
