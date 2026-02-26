# 🔥 HyperFuzz

**High-performance string similarity algorithms implemented in Rust.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg)](https://www.rust-lang.org/)

HyperFuzz is a Python library providing blazing-fast string similarity calculations with **all computations implemented in pure Rust** via PyO3. It offers a RapidFuzz-compatible API alongside advanced algorithms from set-based, alignment, and vector domains — all compiled into a single native extension for maximum performance.

---

## 📦 Installation

```bash
pip install hyperfuzz
```

> **Build from source** (requires Rust toolchain + maturin):
> ```bash
> pip install maturin
> maturin develop --release
> ```

---

## 🚀 Quick Start

```python
from hyperfuzz import distance, fuzz

# Edit distance
dist = distance.levenshtein_distance("kitten", "sitting")       # 3
sim  = distance.levenshtein_normalized_similarity("hello", "hallo")  # 0.8

# Fuzzy matching (0–100 scale)
score = fuzz.ratio("fuzzy wuzzy", "wuzzy fuzzy")                # 45.45
score = fuzz.token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy")     # 100.0
```

---

## 📖 API Reference

HyperFuzz exposes **five algorithm families**. Every function accepts an optional `score_cutoff` keyword argument; if the result is below the cutoff the function returns `0` (or `0.0`). All batch variants accept `pairs: list[tuple[str, str]]` and run on **Rayon thread-pool** for parallel processing.

---

### 1️⃣ `distance` — Edit Distance Metrics

All distance functions live under `hyperfuzz.distance`.

#### Levenshtein

Minimum number of single-character edits (insert, delete, substitute).

```python
from hyperfuzz import distance

# Raw distance (int)
distance.levenshtein_distance("kitten", "sitting")                       # 3
distance.levenshtein_distance("kitten", "sitting", score_cutoff=2)       # 0  (exceeds cutoff)

# Similarity = max_len - distance
distance.levenshtein_similarity("kitten", "sitting")                     # 4

# Normalized (0.0–1.0)
distance.levenshtein_normalized_similarity("kitten", "sitting")          # 0.5714
distance.levenshtein_normalized_distance("kitten", "sitting")            # 0.4286

# Batch — Rayon-parallel
pairs = [("hello", "hallo"), ("foo", "bar")]
distance.levenshtein_distance_batch(pairs)                               # [1, 3]
distance.levenshtein_similarity_batch(pairs)                             # [4, 0]
distance.levenshtein_normalized_distance_batch(pairs)                    # [0.2, 1.0]
distance.levenshtein_normalized_similarity_batch(pairs)                  # [0.8, 0.0]
```

#### Damerau-Levenshtein

Like Levenshtein but also counts **transpositions** of adjacent characters as single edits.

```python
distance.damerau_levenshtein_distance("abcd", "abdc")                    # 1
distance.damerau_levenshtein_similarity("abcd", "abdc")                  # 3
distance.damerau_levenshtein_normalized_distance("abcd", "abdc")         # 0.25
distance.damerau_levenshtein_normalized_similarity("abcd", "abdc")       # 0.75

# Batch
distance.damerau_levenshtein_distance_batch([("abc", "bac")])            # [1]
distance.damerau_levenshtein_normalized_similarity_batch([("abc", "bac")])  # [0.6667]
```

#### OSA (Optimal String Alignment)

Allows transpositions but each substring may only be edited once (no recursive transpositions).

```python
distance.osa_distance("CA", "ABC")                                       # 3
distance.osa_similarity("hello", "helo")                                 # 4
distance.osa_normalized_distance("hello", "helo")                        # 0.2
distance.osa_normalized_similarity("hello", "helo")                      # 0.8

# Batch
distance.osa_distance_batch([("cat", "act")])                            # [2]
distance.osa_normalized_similarity_batch([("cat", "act")])               # [0.3333]
```

#### Hamming

Number of positions where corresponding characters differ. Strings **must** be equal length.

```python
distance.hamming_distance("karolin", "kathrin")                          # 3
distance.hamming_similarity("karolin", "kathrin")                        # 4
distance.hamming_normalized_distance("karolin", "kathrin")               # 0.4286
distance.hamming_normalized_similarity("karolin", "kathrin")             # 0.5714

# Batch
distance.hamming_distance_batch([("abc", "axc")])                        # [1]
distance.hamming_normalized_similarity_batch([("abc", "axc")])           # [0.6667]
```

#### Jaro Similarity

Character matching with transposition penalty, optimized for short strings like names.

```python
distance.jaro_similarity("martha", "marhta")                             # 0.9444
distance.jaro_distance("martha", "marhta")                               # 0.0556
distance.jaro_normalized_similarity("martha", "marhta")                  # 0.9444
distance.jaro_normalized_distance("martha", "marhta")                    # 0.0556

# Batch
distance.jaro_similarity_batch([("martha", "marhta")])                   # [0.9444]
distance.jaro_distance_batch([("abc", "xyz")])                           # [1.0]
```

#### Jaro-Winkler Similarity

Extension of Jaro that gives bonus weight to common **prefixes**.

```python
distance.jaro_winkler_similarity("martha", "marhta")                     # 0.9611
distance.jaro_winkler_distance("martha", "marhta")                       # 0.0389
distance.jaro_winkler_normalized_similarity("martha", "marhta")          # 0.9611
distance.jaro_winkler_normalized_distance("martha", "marhta")            # 0.0389

# Batch
distance.jaro_winkler_similarity_batch([("dwayne", "duane")])            # [0.84]
distance.jaro_winkler_distance_batch([("dwayne", "duane")])              # [0.16]
```

#### Indel (Insertion-Deletion)

Edit distance restricted to insertions and deletions only (no substitutions).

```python
distance.indel_distance("kitten", "sitting")                             # 5
distance.indel_similarity("kitten", "sitting")                           # 8
distance.indel_normalized_distance("kitten", "sitting")                  # 0.3846
distance.indel_normalized_similarity("kitten", "sitting")                # 0.6154

# Batch
distance.indel_distance_batch([("abc", "axc")])                          # [2]
distance.indel_normalized_similarity_batch([("abc", "axc")])             # [0.6667]
```

#### LCSseq (Longest Common Subsequence)

Measures the longest subsequence common to both strings (not necessarily contiguous).

```python
distance.lcs_seq_distance("abcde", "ace")                                # 2
distance.lcs_seq_similarity("abcde", "ace")                              # 3
distance.lcs_seq_normalized_distance("abcde", "ace")                     # 0.4
distance.lcs_seq_normalized_similarity("abcde", "ace")                   # 0.6

# Batch
distance.lcs_seq_distance_batch([("abcde", "ace")])                      # [2]
distance.lcs_seq_normalized_similarity_batch([("abcde", "ace")])         # [0.6]
```

#### LCSstr (Longest Common Substring)

Measures the longest **contiguous** substring common to both strings.

```python
distance.lcs_str_similarity("abcxyz", "xyzabc")                         # 3
distance.lcs_str_distance("abcxyz", "xyzabc")                           # 3
distance.lcs_str_normalized_similarity("abcxyz", "xyzabc")              # 0.5
distance.lcs_str_normalized_distance("abcxyz", "xyzabc")                # 0.5

# Batch
distance.lcs_str_similarity_batch([("abcxyz", "xyzabc")])               # [3]
distance.lcs_str_normalized_similarity_batch([("abcxyz", "xyzabc")])    # [0.5]
```

#### Prefix

Counts how many leading characters are shared.

```python
distance.prefix_similarity("hello world", "hello there")                 # 6
distance.prefix_distance("hello world", "hello there")                   # 5
distance.prefix_normalized_similarity("hello world", "hello there")      # 0.5455
distance.prefix_normalized_distance("hello world", "hello there")        # 0.4545

# Batch
distance.prefix_similarity_batch([("abc", "abd")])                       # [2]
distance.prefix_normalized_similarity_batch([("abc", "abd")])            # [0.6667]
```

#### Postfix

Counts how many trailing characters are shared.

```python
distance.postfix_similarity("hello world", "cruel world")                # 6
distance.postfix_distance("hello world", "cruel world")                  # 5
distance.postfix_normalized_similarity("hello world", "cruel world")     # 0.5455
distance.postfix_normalized_distance("hello world", "cruel world")       # 0.4545

# Batch
distance.postfix_similarity_batch([("testing", "running")])              # [3]
distance.postfix_normalized_similarity_batch([("testing", "running")])   # [0.4286]
```

---

### 2️⃣ `fuzz` — Fuzzy String Matching

All fuzz functions return scores on a **0–100 scale** and live under `hyperfuzz.fuzz`.

#### `fuzz.ratio`

Simple full-string ratio based on LCS. Compares the full strings character-by-character.

```python
from hyperfuzz import fuzz

fuzz.ratio("this is a test", "this is a test!")                           # 96.55
fuzz.ratio("fuzzy wuzzy", "wuzzy fuzzy")                                 # 45.45
fuzz.ratio("hello", "hello")                                             # 100.0

# With cutoff — returns 0.0 if result < cutoff
fuzz.ratio("abc", "xyz", score_cutoff=50.0)                              # 0.0

# Batch
fuzz.ratio_batch([("hello", "hallo"), ("abc", "abc")])                   # [80.0, 100.0]
```

#### `fuzz.partial_ratio`

Finds the best matching **substring** of the longer string to the shorter one. Ideal for detecting when one string is contained within another.

```python
fuzz.partial_ratio("test", "this is a test string")                      # 100.0
fuzz.partial_ratio("fuzzy wuzzy", "wuzzy fuzzy bear")                    # 62.5
fuzz.partial_ratio("hello", "hello world")                               # 100.0

# Batch
fuzz.partial_ratio_batch([("test", "this is a test")])                   # [100.0]
```

#### `fuzz.token_sort_ratio`

Tokenizes both strings, **sorts** the tokens alphabetically, then computes ratio. Handles word-order differences.

```python
fuzz.token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy")                      # 100.0
fuzz.token_sort_ratio("New York Mets", "Mets New York")                  # 100.0
fuzz.token_sort_ratio("hello world", "world hello there")                # 76.92

# Batch
fuzz.token_sort_ratio_batch([("a b c", "c b a")])                        # [100.0]
```

#### `fuzz.token_set_ratio`

Splits tokens into intersection and remainder sets, then computes the best ratio among combinations. Robust to **repeated and extra words**.

```python
fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")       # 100.0
fuzz.token_set_ratio("hello world", "world peace")                       # 50.0
fuzz.token_set_ratio("New York Mets", "Mets York New")                   # 100.0

# Batch
fuzz.token_set_ratio_batch([("a b", "b a c")])                           # [100.0]
```

#### `fuzz.token_ratio`

Combines `token_sort_ratio` and `token_set_ratio`, returning the best score. Handles both word-order and extra-word scenarios.

```python
fuzz.token_ratio("fuzzy wuzzy", "wuzzy fuzzy")                           # 100.0
fuzz.token_ratio("hello world", "world hello friend")                    # 100.0

# Batch
fuzz.token_ratio_batch([("a b", "b a")])                                 # [100.0]
```

#### `fuzz.partial_token_sort_ratio`

Sorts tokens then applies **partial_ratio** (substring matching). Best for cases with word-order and substring variations.

```python
fuzz.partial_token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy")   # 100.0
fuzz.partial_token_sort_ratio("hello world foo", "foo world hello")      # 100.0

# Batch
fuzz.partial_token_sort_ratio_batch([("a b c", "c b")])                  # [100.0]
```

#### `fuzz.partial_token_set_ratio`

Set-based + partial matching. Returns **100** when any token is shared (common word optimization).

```python
fuzz.partial_token_set_ratio("hello world", "world peace")               # 100.0
fuzz.partial_token_set_ratio("kitten", "sitting")                        # 66.67
fuzz.partial_token_set_ratio("abc def", "def ghi")                       # 100.0

# Batch
fuzz.partial_token_set_ratio_batch([("hello world", "earth world")])     # [100.0]
```

#### `fuzz.partial_token_ratio`

Combines partial_token_sort + partial_token_set, returning the best score.

```python
fuzz.partial_token_ratio("hello world", "world peace hello")             # 100.0
fuzz.partial_token_ratio("abc", "xyz")                                   # 0.0

# Batch
fuzz.partial_token_ratio_batch([("hello", "hello world extra")])         # [100.0]
```

#### `fuzz.WRatio` (Weighted Ratio)

Automatically selects the best algorithm based on string length ratio with weighted scores. The recommended default for general-purpose fuzzy matching.

```python
fuzz.WRatio("this is a test", "this is a test!")                          # 96.55
fuzz.WRatio("test", "this is a long test string")                        # 90.0
fuzz.WRatio("hello", "hello")                                            # 100.0

# Batch
fuzz.WRatio_batch([("hello", "hallo"), ("test", "testing 123")])         # [80.0, 85.5]
```

#### `fuzz.QRatio` (Quick Ratio)

Same as `ratio` but returns `0.0` for empty strings. Fastest fuzz metric.

```python
fuzz.QRatio("this is a test", "this is a test!")                          # 96.55
fuzz.QRatio("", "test")                                                   # 0.0
fuzz.QRatio("hello", "hello")                                            # 100.0

# Batch
fuzz.QRatio_batch([("hello", "hallo"), ("", "test")])                    # [80.0, 0.0]
```

---

### 3️⃣ Set-Based Algorithms

Token (word) level similarity using multiset (bag) semantics. All are available directly on the `hyperfuzz` module.

```python
from hyperfuzz import (
    jaccard_similarity, jaccard_distance,
    sorensen_dice_similarity, sorensen_dice_distance,
    tversky_similarity, tversky_distance,
    overlap_similarity, overlap_distance,
)
```

#### Jaccard Similarity

`|A ∩ B| / |A ∪ B|` — ratio of shared words to total unique words.

```python
jaccard_similarity("hello world foo", "hello world bar")                 # 0.5
jaccard_distance("hello world foo", "hello world bar")                   # 0.5

# Batch
from hyperfuzz import jaccard_similarity_batch
jaccard_similarity_batch([("a b c", "b c d")])                           # [0.5]
```

#### Sørensen-Dice Coefficient

`2|A ∩ B| / (|A| + |B|)` — harmonic mean of shared words.

```python
sorensen_dice_similarity("hello world foo", "hello world bar")           # 0.6667
sorensen_dice_distance("hello world foo", "hello world bar")             # 0.3333

# Batch
from hyperfuzz import sorensen_dice_similarity_batch
sorensen_dice_similarity_batch([("a b c", "b c d")])                     # [0.6667]
```

#### Tversky Index

`|A ∩ B| / (|A ∩ B| + α|A - B| + β|B - A|)` — generalizes Jaccard (α=1, β=1) and Dice (α=0.5, β=0.5).

```python
tversky_similarity("hello world foo", "hello world bar", alpha=1.0, beta=1.0)   # Jaccard
tversky_similarity("hello world foo", "hello world bar", alpha=0.5, beta=0.5)   # Dice
tversky_distance("hello world foo", "hello world bar", alpha=1.0, beta=1.0)     # 0.5

# Batch
from hyperfuzz import tversky_similarity_batch
tversky_similarity_batch([("a b", "b c")], alpha=1.0, beta=1.0)         # [0.3333]
```

#### Overlap Coefficient

`|A ∩ B| / min(|A|, |B|)` — measures whether one set is a subset of another.

```python
overlap_similarity("hello world", "hello world foo bar")                 # 1.0
overlap_distance("hello world", "hello world foo bar")                   # 0.0

# Batch
from hyperfuzz import overlap_similarity_batch
overlap_similarity_batch([("a b", "a b c d")])                           # [1.0]
```

---

### 4️⃣ Alignment Algorithms

Bioinformatics-inspired sequence alignment. Default scoring: `match=1, mismatch=0, gap_cost=1`.

```python
from hyperfuzz import (
    smith_waterman_score, smith_waterman_normalized_similarity,
    needleman_wunsch_score, needleman_wunsch_normalized_similarity,
)
```

#### Smith-Waterman (Local Alignment)

Finds the best **local** alignment sub-region.

```python
smith_waterman_score("ACGT", "ACCT", match_score=1.0, mismatch_score=0.0, gap_cost=1.0)
# Returns the local alignment score

smith_waterman_normalized_similarity(
    "hello world", "world",
    match_score=1.0, mismatch_score=0.0, gap_cost=1.0
)  # 0.0–1.0

# Batch
from hyperfuzz import smith_waterman_score_batch
smith_waterman_score_batch(
    [("abc", "abc"), ("abc", "xyz")],
    match_score=1.0, mismatch_score=0.0, gap_cost=1.0
)
```

#### Needleman-Wunsch (Global Alignment)

Optimal **global** alignment end-to-end.

```python
needleman_wunsch_score("ACGT", "ACCT", match_score=1.0, mismatch_score=0.0, gap_cost=1.0)
# Returns the global alignment score

needleman_wunsch_normalized_similarity(
    "hello", "hallo",
    match_score=1.0, mismatch_score=0.0, gap_cost=1.0
)  # 0.0–1.0

# Batch
from hyperfuzz import needleman_wunsch_score_batch
needleman_wunsch_score_batch(
    [("abc", "abc"), ("abc", "xyz")],
    match_score=1.0, mismatch_score=0.0, gap_cost=1.0
)
```

---

### 5️⃣ Vector / Statistical Methods

Token or n-gram frequency based similarity.

```python
from hyperfuzz import (
    cosine_similarity, cosine_distance,
    soft_tfidf_similarity, soft_tfidf_distance,
)
```

#### Cosine Similarity

Computes cosine similarity between term-frequency vectors. Supports **word-based** (default) and **n-gram** modes.

```python
# Word-based (default, TextDistance compatible)
cosine_similarity("hello world", "hello there", use_words=True, ngram_size=2)   # 0.5

# N-gram based
cosine_similarity("night", "nacht", use_words=False, ngram_size=2)              # character 2-grams

cosine_distance("hello world", "hello there", use_words=True, ngram_size=2)     # 0.5

# Batch
from hyperfuzz import cosine_similarity_batch
cosine_similarity_batch(
    [("hello world", "hello there")],
    use_words=True, ngram_size=2
)
```

#### Soft-TFIDF

Combines TF-IDF weighting with fuzzy token matching (Jaro-Winkler). Tokens are considered a match if their similarity ≥ `threshold`.

```python
soft_tfidf_similarity("hello world", "helo wrld", threshold=0.8)               # > 0.0
soft_tfidf_distance("hello world", "helo wrld", threshold=0.8)                 # < 1.0

# Batch
from hyperfuzz import soft_tfidf_similarity_batch
soft_tfidf_similarity_batch(
    [("New York", "New Yrok")],
    threshold=0.8
)
```

---

## ⚡ Performance: HyperFuzz vs RapidFuzz

All benchmarks run with **10,000 iterations** per test case on Python 3.13.

### Fuzz — Short Text

Average speedup per algorithm on short text pairs (names, IDs, emails; < 25 chars):

| Algorithm | Avg Speedup |
|-----------|-------------|
| `fuzz.ratio` | **2.20x** |
| `fuzz.partial_ratio` | **2.88x** |
| `fuzz.token_sort_ratio` | **3.22x** |
| `fuzz.token_set_ratio` | **3.13x** |
| `fuzz.token_ratio` | **2.86x** |
| `fuzz.partial_token_sort_ratio` | **4.26x** |
| `fuzz.partial_token_set_ratio` | **2.64x** |
| `fuzz.partial_token_ratio` | **2.73x** |
| `fuzz.WRatio` | **2.01x** |
| `fuzz.QRatio` | **2.18x** |

### Fuzz — Long Text

Average speedup per algorithm on long text pairs (paragraphs; 60–220 chars):

| Algorithm | Avg Speedup |
|-----------|-------------|
| `fuzz.ratio` | **1.19x** |
| `fuzz.partial_ratio` | **2.67x** |
| `fuzz.token_sort_ratio` | **1.29x** |
| `fuzz.token_set_ratio` | **1.24x** |
| `fuzz.token_ratio` | **1.10x** |
| `fuzz.partial_token_sort_ratio` | **2.64x** |
| `fuzz.partial_token_set_ratio` | **1.97x** |
| `fuzz.partial_token_ratio` | **2.09x** |
| `fuzz.WRatio` | **1.06x** |
| `fuzz.QRatio` | **1.16x** |

### Overall

| Metric | Value |
|--------|-------|
| Score Accuracy | **98.8%** (249/252) |
| HyperFuzz Faster | **81.3%** of test cases |
| **Avg Speedup (Short Text)** | **2.82x** 🚀 |
| **Avg Speedup (Long Text)** | **1.64x** 🚀 |
| **Overall Fuzz Avg Speedup** | **2.43x** 🚀 |

> Benchmark source: 252 test cases covering 8 short + 4 long text pair scenarios across 21 algorithms.

---

## 🏗️ Architecture

```
hyperfuzz
├── distance               # Edit distance metrics
│   ├── levenshtein        #   Myers' bit-parallel O(n·m/64)
│   ├── damerau_levenshtein#   Full DL with transpositions
│   ├── osa                #   Optimal String Alignment
│   ├── hamming            #   Position-wise comparison
│   ├── jaro               #   Character-matching metric
│   ├── jaro_winkler       #   Prefix-weighted Jaro
│   ├── indel              #   Insert/Delete only
│   ├── lcs_seq            #   Longest Common Subsequence
│   ├── lcs_str            #   Longest Common Substring
│   ├── prefix             #   Common prefix length
│   └── postfix            #   Common suffix length
├── fuzz                   # Fuzzy matching (0–100 scale)
│   ├── ratio              #   Full-string LCS ratio
│   ├── partial_ratio      #   Best-substring ratio
│   ├── token_sort_ratio   #   Sorted-token ratio
│   ├── token_set_ratio    #   Set-intersection ratio
│   ├── token_ratio        #   Combined sort+set
│   ├── partial_token_*    #   Partial variants
│   ├── WRatio             #   Weighted auto-select
│   └── QRatio             #   Quick ratio
├── set_based              # Jaccard, Dice, Tversky, Overlap
├── alignment              # Smith-Waterman, Needleman-Wunsch
└── vector                 # Cosine similarity, Soft-TFIDF
```

**Key Rust Optimizations:**
- 🧬 **Myers' bit-parallel** — O(n·m/64) Levenshtein via 64-bit SIMD-style word operations
- 🔄 **Thread-local buffers** — zero-allocation LCS via `thread_local!` reusable buffers
- ⚡ **Rayon parallelism** — all `_batch` operations auto-distribute across CPU cores
- 🏎️ **ASCII fast paths** — byte-level operations bypass expensive Unicode iteration
- 📦 **SmallVec / ahash** — stack-allocated token vectors and fast hash maps

---

## 📋 Complete Function List

<details>
<summary><strong>distance module</strong> (66 functions)</summary>

| Family | Functions |
|--------|-----------|
| Levenshtein | `levenshtein_distance`, `levenshtein_similarity`, `levenshtein_normalized_distance`, `levenshtein_normalized_similarity`, + 4 batch |
| Damerau-Levenshtein | `damerau_levenshtein_distance`, `damerau_levenshtein_similarity`, `damerau_levenshtein_normalized_distance`, `damerau_levenshtein_normalized_similarity`, + 2 batch |
| OSA | `osa_distance`, `osa_similarity`, `osa_normalized_distance`, `osa_normalized_similarity`, + 2 batch |
| Hamming | `hamming_distance`, `hamming_similarity`, `hamming_normalized_distance`, `hamming_normalized_similarity`, + 2 batch |
| Jaro | `jaro_similarity`, `jaro_distance`, `jaro_normalized_similarity`, `jaro_normalized_distance`, + 2 batch |
| Jaro-Winkler | `jaro_winkler_similarity`, `jaro_winkler_distance`, `jaro_winkler_normalized_similarity`, `jaro_winkler_normalized_distance`, + 2 batch |
| Indel | `indel_distance`, `indel_similarity`, `indel_normalized_distance`, `indel_normalized_similarity`, + 2 batch |
| LCSseq | `lcs_seq_distance`, `lcs_seq_similarity`, `lcs_seq_normalized_distance`, `lcs_seq_normalized_similarity`, + 2 batch |
| LCSstr | `lcs_str_similarity`, `lcs_str_distance`, `lcs_str_normalized_similarity`, `lcs_str_normalized_distance`, + 2 batch |
| Prefix | `prefix_distance`, `prefix_similarity`, `prefix_normalized_distance`, `prefix_normalized_similarity`, + 2 batch |
| Postfix | `postfix_distance`, `postfix_similarity`, `postfix_normalized_distance`, `postfix_normalized_similarity`, + 2 batch |

</details>

<details>
<summary><strong>fuzz module</strong> (20 functions)</summary>

| Function | Batch |
|----------|-------|
| `ratio` | `ratio_batch` |
| `partial_ratio` | `partial_ratio_batch` |
| `token_sort_ratio` | `token_sort_ratio_batch` |
| `token_set_ratio` | `token_set_ratio_batch` |
| `token_ratio` | `token_ratio_batch` |
| `partial_token_sort_ratio` | `partial_token_sort_ratio_batch` |
| `partial_token_set_ratio` | `partial_token_set_ratio_batch` |
| `partial_token_ratio` | `partial_token_ratio_batch` |
| `WRatio` | `WRatio_batch` |
| `QRatio` | `QRatio_batch` |

</details>

<details>
<summary><strong>Set-based</strong> (12 functions)</summary>

`jaccard_similarity`, `jaccard_distance`, `jaccard_similarity_batch`,
`sorensen_dice_similarity`, `sorensen_dice_distance`, `sorensen_dice_similarity_batch`,
`tversky_similarity`, `tversky_distance`, `tversky_similarity_batch`,
`overlap_similarity`, `overlap_distance`, `overlap_similarity_batch`

</details>

<details>
<summary><strong>Alignment</strong> (6 functions)</summary>

`smith_waterman_score`, `smith_waterman_normalized_similarity`, `smith_waterman_score_batch`,
`needleman_wunsch_score`, `needleman_wunsch_normalized_similarity`, `needleman_wunsch_score_batch`

</details>

<details>
<summary><strong>Vector</strong> (6 functions)</summary>

`cosine_similarity`, `cosine_distance`, `cosine_similarity_batch`,
`soft_tfidf_similarity`, `soft_tfidf_distance`, `soft_tfidf_similarity_batch`

</details>

---

## License

MIT License
