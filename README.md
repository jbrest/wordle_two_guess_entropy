# Wordle Non-Adaptive Entropy Optimizer

This project explores non-adaptive Wordle strategies for 1, 2, and 3 fixed opening guesses:

**What fixed opening guesses maximize information about the hidden word, assuming you do NOT adapt subsequent guesses based on prior feedback?**

Most Wordle solvers focus on *adaptive* play (recomputing the best guess each turn). This project instead solves a different problem:

> Choose multiple words in advance that, together, maximally reduce uncertainty about the answer.

---

## 🚀 What This Project Does

- Loads Wordle answer and allowed word lists
- Computes Wordle feedback patterns efficiently
- Builds a reusable pattern matrix for fast evaluation
- Computes entropy for single guesses
- Computes **joint entropy** for two-word and three-word non-adaptive opening sequences
- Uses provably safe pruning bounds to avoid most of the naive \(O(n^2)\) and \(O(n^3)\) work

The result is a ranked list of the best *non-adaptive opening sequences* (1, 2, or 3 words).

---

## 🧠 Why This Is Interesting

Instead of simulating perfect play, this project designs a human-friendly strategy.

Even after a high-information first guess, the information-theoretically strongest second guesses are often surprisingly unintuitive. They may use letter combinations that feel odd or unhelpful to a human player, even though they are mathematically efficient at partitioning the space of possible answers.

The goal of this project is to generate a high-information two-word “probe” that works well across the entire answer space. After that probe — or earlier, if the feedback from the first word is especially suggestive — the player can switch to normal, human-style adaptive reasoning once the remaining possibilities are sufficiently narrow to grasp intuitively.

In other words, the purpose of this project is to front-load as much information as possible into the first two moves, when human intuition is weakest, and then hand the problem back to the player once it becomes cognitively manageable.

---

## 🛠 Installation

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd <your-repo-directory>
pip install -r requirements.txt
```

---

## 📁 Word Lists

Place word lists in:

```
data/answers.txt
data/allowed.txt
```

You may use the original Wordle lists or expanded lists. Results depend on the answer set size.

---

## 🧮 First Run

On first run, the program builds a pattern matrix:

```bash
python wordle_entropy.py
```

This may take a few minutes and creates:

```
data/pattern_matrix.npy
```

Future runs reuse this file.

---

## 🔍 Find Best Two-Guess Openings

```bash
python wordle_entropy.py -words 2
```

This runs the optimized search and prints the top-scoring word pairs.

For verbose pair output with individual entropies and first-word cost:

```bash
python wordle_entropy.py -words 2 -verbose
```

To evaluate one specific pair directly (overrides `-words`):

```bash
python wordle_entropy.py -pair raise mount
```

Verbose works with `-pair` as well:

```bash
python wordle_entropy.py -pair raise mount -verbose
```

To evaluate one specific triple directly (overrides `-words`):

```bash
python wordle_entropy.py -triple raise mount clint
```

Verbose works with `-triple` as well:

```bash
python wordle_entropy.py -triple raise mount clint -verbose
```

To search for top three-word non-adaptive openings:

```bash
python wordle_entropy.py -words 3
```

This mode is extremely expensive and asks for confirmation before starting.

**Important:** The 3-word search uses **strict pruning** that guarantees finding the exact top 50 triples. It will never miss a triple that belongs in the top 50, but the search may take days to weeks depending on hardware.

For unattended runs, skip confirmation with:

```bash
python wordle_entropy.py -words 3 -force
```

For multi-day runs with live dashboard + diagnostics:

```bash
python wordle_entropy.py -words 3 -force -progress dashboard -debug-prune-file /tmp/p2_diag.csv
```

### Long-Run Controls (`-words 3`)

- `-workers N`: number of worker processes (default: CPU count)
- `-chunk-size N`: in-flight task multiplier per worker (default: `2`)
- `-progress dashboard|live|log|off`: progress display mode
- `-history-seconds N`: sampling period for history summaries
- `-stats-file PATH`: sampled runtime/pruning CSV output
- `-debug-prune-file PATH`: line-buffered prune diagnostics CSV

Checkpointing is enabled by default for 3-word runs:

- Default checkpoint file: `.wordle3_checkpoint.json`
- Autosaves at startup and after each completed outer-loop iteration
- Preserves all progress including elapsed runtime across restarts
- On restart, existing checkpoint behavior is controlled with:
  - `-resume ask` (default): prompt `Resume? [Y/n]`
  - `-resume auto`: always resume
  - `-resume new`: ignore old checkpoint and start fresh

Example fresh restart (ignore saved state):

```bash
python wordle_entropy.py -words 3 -force -progress dashboard -resume new
```

Resume from existing checkpoint (no prompt):

```bash
python wordle_entropy.py -words 3 -force -progress dashboard -resume auto
```

Pair order is an artifact of the search implementation: pairs are emitted as
`first + second` with the first word having greater-than-or-equal single-word
entropy than the second. In practice, this front-loads information into guess 1
and improves your chance of a dynamic offramp after the first guess. It is
still a pair-optimization result, so the first word is not guaranteed to match
the globally optimal standalone single opener.

Output lines include answer-membership flags in brackets:

- `[++]` both words are valid hidden-answer words (`answers.txt`)
- `[+-]` only the first word is an answer word
- `[-+]` only the second word is an answer word
- `[--]` neither word is an answer word (both are still valid guesses)

For `python wordle_entropy.py` single-guess output:

- `[+]` the word is in `answers.txt`
- `[-]` the word is guess-only (in `allowed.txt` but not `answers.txt`)

Single-guess mode is the default and equivalent to:

```bash
python wordle_entropy.py -words 1
```

---

## ⚙️ Optimization Techniques Used

The project uses several safe, provable optimizations:

- **Precomputed pattern matrix**: All guess/answer feedback patterns computed once
- **Monotonic upper bounds**: Skip entire suffixes when upper bounds fall below floor
- **Non-monotonic bounds**: Individual pruning when bounds are not monotonically decreasing
- **Strict pruning (3-word mode)**: Guarantees finding exact top 50, no false negatives
- **Parallel processing**: Multi-core worker pool for 3-word search
- **Shared floor optimization**: Workers share best-known floor to prune aggressively

For 3-word search:
- Pre-H12 pruning uses monotonic `h1 + h2 + h3_best` bound (can prune entire suffix)
- Post-H12 pruning uses non-monotonic `h12 + h3_best` bound (prunes individual j only)
- Inner loop uses monotonic `h12 + h3` bound (can prune suffix)

These reduce effective computation from trillions of triples to a manageable fraction while guaranteeing completeness.

---

## 📊 Interpreting the "Bits"

Entropy values are measured in **bits of information gained**.

The maximum possible information depends on the number of possible answers:

```
max_bits = log2(number_of_answers)
```

So entropy values are meaningful *within a given answer list*, but not directly comparable across different lists.

---

## 📜 License

MIT License. See LICENSE file.

---

## 🧾 Version History

Project version history is tracked in `CHANGELOG.md`.

---

## 🤖 Attribution

This project was developed with design and optimization guidance from OpenAI's ChatGPT.

The core idea and problem framing were developed by the repository author.
