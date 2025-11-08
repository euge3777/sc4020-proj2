# Pattern Mining Usage Guide

## Customizable Parameters

You can now control the following parameters in the GSP algorithm:

### 1. `min_support` (float, default: 0.15)

- Minimum support threshold for patterns
- Range: 0.0 to 1.0
- Lower values = more patterns (but potentially less meaningful)
- Higher values = fewer, more frequent patterns

### 2. `max_pattern_length` (int, default: 5)

- Maximum length of sequential patterns to mine
- Example: `max_pattern_length=3` will mine patterns up to 3 elements long
- Higher values = longer patterns (but more computation time)

### 3. `max_gap` (int, default: 1)

- Maximum gap allowed between consecutive pattern elements
- `max_gap=0`: Elements must be adjacent (no items between them)
- `max_gap=1`: At most 1 item can appear between pattern elements
- `max_gap=2`: At most 2 items can appear between pattern elements

### 4. `top_k` (int, default: 15)

- Number of top patterns to display for each diagnosis class
- Only affects display, not the mining process
- All patterns are still mined and used for classification

## Usage Examples

### Example 1: Default Settings

```python
results = run_pattern_mining_analysis()
```

### Example 2: Stricter Patterns (higher support, shorter length)

```python
results = run_pattern_mining_analysis(
    min_support=0.3,         # Only patterns in 30%+ of sequences
    max_pattern_length=3,    # Maximum 3 elements
    max_gap=0,              # Elements must be adjacent
    top_k=10                # Show top 10 patterns
)
```

### Example 3: More Flexible Patterns (lower support, longer sequences)

```python
results = run_pattern_mining_analysis(
    min_support=0.1,         # Patterns in 10%+ of sequences
    max_pattern_length=7,    # Maximum 7 elements
    max_gap=2,              # Allow 2 items between elements
    top_k=20                # Show top 20 patterns
)
```

### Example 4: Very Strict Patterns

```python
results = run_pattern_mining_analysis(
    min_support=0.5,         # Only very frequent patterns
    max_pattern_length=2,    # Only 2-element patterns
    max_gap=0,              # Must be consecutive
    top_k=5                 # Show top 5 only
)
```

## How to Run

### Option 1: Edit the file directly

Open `pattern_mining.py` and modify the parameters in the `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    results = run_pattern_mining_analysis(
        min_support=0.2,        # Your desired value
        max_pattern_length=4,   # Your desired value
        max_gap=1,             # Your desired value
        top_k=10               # Your desired value
    )
```

Then run:

```bash
python pattern_mining.py
```

### Option 2: Import and call from another script

```python
from pattern_mining import run_pattern_mining_analysis

# Run with custom parameters
results = run_pattern_mining_analysis(
    min_support=0.2,
    max_pattern_length=4,
    max_gap=1,
    top_k=10
)
```

### Option 3: Use the class directly for more control

```python
from pattern_mining import SequentialPatternMiner
import pandas as pd

# Load your data
df = pd.read_csv('data/Cancer_Data_sequences_quantile.csv')

# Create miner with custom parameters
miner = SequentialPatternMiner(
    min_support=0.2,
    max_pattern_length=4,
    max_gap=2
)

# Mine patterns
patterns = miner.mine_patterns(
    df['sequence'].tolist(),
    df['diagnosis'].tolist()
)

# Access patterns
malignant_patterns = patterns['Malignant']['sequential_patterns']
benign_patterns = patterns['Benign']['sequential_patterns']
```

## Understanding max_gap

### max_gap = 0 (Strict Consecutive)

```
Sequence: [A, B, C, D, E]
Pattern: (A, B, C) ✓ - consecutive
Pattern: (A, C, E) ✗ - not consecutive
```

### max_gap = 1 (One Gap Allowed)

```
Sequence: [A, B, C, D, E]
Pattern: (A, B, C) ✓ - consecutive
Pattern: (A, C, E) ✓ - gaps of 1 between each element
Pattern: (A, D)    ✓ - gap of 2 between A and D
Pattern: (A, E)    ✗ - gap of 3, exceeds max_gap=1
```

### max_gap = 2 (Two Gaps Allowed)

```
Sequence: [A, B, C, D, E]
Pattern: (A, E)    ✓ - gap of 3 is allowed
Pattern: (A, C)    ✓ - gap of 1 is allowed
```

## Output Information

The program will display:

1. Configuration summary showing all parameter values
2. For each discretization method (quantile, uniform, kmeans):
   - Top-K patterns for Malignant diagnosis
   - Top-K patterns for Benign diagnosis
   - Pattern statistics (total patterns, max length)
   - Classification accuracy using pattern features
   - Number of features generated

## Performance Considerations

- **Lower `min_support`** → More patterns → Longer computation time
- **Higher `max_pattern_length`** → More candidates → Longer computation time
- **Higher `max_gap`** → More flexible matching → Longer computation time
- **`top_k`** → Only affects display, not performance

## Recommended Starting Points

For **exploratory analysis**:

```python
min_support=0.1, max_pattern_length=5, max_gap=1, top_k=20
```

For **reliable patterns**:

```python
min_support=0.2, max_pattern_length=4, max_gap=1, top_k=15
```

For **very strict patterns**:

```python
min_support=0.3, max_pattern_length=3, max_gap=0, top_k=10
```
