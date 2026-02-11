# Wordle Two-Guess Entropy Optimizer

This project explores a non-adaptive Wordle strategy:

**What are the two fixed opening guesses that maximize information about the hidden word, assuming you do NOT adapt your second guess based on feedback from the first?**

Most Wordle solvers focus on *adaptive* play (recomputing the best guess each turn). This project instead solves a different problem:

> Choose two words in advance that, together, maximally reduce uncertainty about the answer.

---

## 🚀 What This Project Does

- Loads Wordle answer and allowed word lists
- Computes Wordle feedback patterns efficiently
- Builds a reusable pattern matrix for fast evaluation
- Computes entropy for single guesses
- Computes **joint entropy** for all two-word opening pairs
- Uses provably safe pruning bounds to avoid most of the naive \(O(n^2)\) work

The result is a ranked list of the best *non-adaptive two-guess openings*.

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
For unattended runs, skip confirmation with:

```bash
python wordle_entropy.py -words 3 -force
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

- Precomputed guess/answer pattern matrix
- Entropy upper bounds to skip hopeless first guesses
- Pairwise bounds to skip weak second guesses
- Symmetry reduction (word A + B is same as B + A)

These reduce effective computation from billions of pairs to a tiny fraction.

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
