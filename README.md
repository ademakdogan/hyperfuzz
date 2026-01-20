# 🔥 HyperFuzz

**High-performance string similarity algorithms implemented in Rust.**

> 🚧 **Coming Soon** - This project is under active development.

## Overview

HyperFuzz is a Python library providing blazing-fast string similarity calculations with all computations implemented in pure Rust using PyO3.

## Features (Planned)

- 🚀 **All computations in Rust** - Maximum performance
- 📦 **Drop-in replacement** - Compatible API with RapidFuzz
- 🔄 **Batch operations** - Parallel processing with Rayon
- 📊 **Comprehensive algorithms** - Distance metrics, fuzz matching, and more

## Installation

```bash
pip install hyperfuzz
```

## Quick Start

```python
from hyperfuzz import distance, fuzz

# Distance metrics
dist = distance.levenshtein_distance("kitten", "sitting")
sim = distance.levenshtein_normalized_similarity("hello", "hallo")

# Fuzz matching
score = fuzz.ratio("fuzzy wuzzy", "wuzzy fuzzy")
```

## License

MIT License
