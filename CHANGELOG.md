# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-02-10

### Added
- Unified CLI entrypoint: `wordle_entropy.py`.
- New CLI options:
  - `-words {1,2,3}` (default: `1`)
  - `-verbose` (two-word and three-word outputs)
  - `-pair WORD1 WORD2` (overrides `-words`, supports `-verbose`)
  - `-triple WORD1 WORD2 WORD3` (overrides `-words`, supports `-verbose`)
  - `-force` (skip confirmation prompt for `-words 3`)
- GitHub Actions CI workflow for syntax checks in `.github/workflows/ci.yml`.
- Cache staleness guard for pattern matrix using timestamp checks against
  `data/answers.txt` and `data/allowed.txt`.

### Changed
- Data file path handling now resolves relative to source files, not the shell
  working directory.
- `README.md` updated for unified CLI usage and output conventions.
- MIT license text completed.

### Removed
- Legacy entrypoints `main.py` and `two_guess_main.py` (superseded by
  `wordle_entropy.py`).

## [0.1.0] - 2026-02-09

### Added
- Initial release:
  - Word list loading
  - Wordle feedback pattern encoding
  - Pattern matrix build/load cache
  - Single-guess entropy ranking
  - Non-adaptive two-guess joint-entropy search with pruning
