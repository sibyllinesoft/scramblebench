# ScrambleBench CLI Guide

The ScrambleBench CLI provides comprehensive command-line interface capabilities for language generation, text transformation, and benchmark processing.

## Installation

```bash
# Install base dependencies
pip install -e .

# Install with NLP capabilities (recommended for better text processing)
pip install -e ".[nlp]"

# Install development dependencies
pip install -e ".[dev]"
```

## Command Structure

The CLI is organized into command groups:

- `language` - Language generation and management
- `batch` - Batch processing of benchmark data
- `transform` - Text transformation operations
- `util` - Utility commands for analysis and export

## Global Options

All commands support these global options:

```bash
--verbose, -v          Enable verbose output
--quiet, -q           Suppress non-essential output
--output-format       Output format: text, json, yaml (default: text)
--data-dir            Base data directory (default: data)
```

## Language Generation Commands

### Generate a New Language

```bash
# Generate a substitution language
scramblebench language generate mylang --type substitution --complexity 5

# Generate with specific parameters
scramblebench language generate fantasy \
  --type phonetic \
  --complexity 7 \
  --vocab-size 2000 \
  --seed 42

# Generate without saving
scramblebench language generate temp --type scrambled --no-save
```

**Language Types:**
- `substitution` - Simple character/word substitution
- `phonetic` - Phonetically plausible transformations  
- `scrambled` - Character scrambling with rules
- `synthetic` - Generated vocabulary with grammar

**Complexity Levels:** 1-10 (higher = more complex transformation rules)

### List Available Languages

```bash
# Table view (default)
scramblebench language list

# Simple list
scramblebench language list --format simple

# JSON output
scramblebench language list --format json
```

### Show Language Details

```bash
# Basic information
scramblebench language show mylang

# Show transformation rules
scramblebench language show mylang --show-rules --limit 50

# Show vocabulary mappings
scramblebench language show mylang --show-vocabulary --limit 100

# Complete details
scramblebench language show mylang --show-rules --show-vocabulary
```

### Delete a Language

```bash
# With confirmation
scramblebench language delete mylang

# Force deletion
scramblebench language delete mylang --force
```

## Batch Processing Commands

### Extract Vocabulary from Benchmark Files

```bash
# Extract from CSV file
scramblebench batch extract-vocab data/benchmarks/math_problems.csv

# With custom parameters
scramblebench batch extract-vocab questions.json \
  --output custom_vocab.json \
  --min-freq 3 \
  --max-words 10000
```

### Transform Benchmark Problems

```bash
# Transform using existing language
scramblebench batch transform questions.json mylang

# With custom settings
scramblebench batch transform data/benchmarks/qa.json fantasy \
  --output transformed_qa.json \
  --transform-numbers \
  --transform-proper-nouns \
  --batch-size 50
```

## Text Transformation Commands

### Transform Single Text

```bash
# Transform text using a language
scramblebench transform text "Hello, how are you?" mylang

# Preserve different elements
scramblebench transform text "John has 5 apples" mylang \
  --preserve-numbers \
  --preserve-proper-nouns
```

### Proper Noun Replacement

```bash
# Random replacement
scramblebench transform proper-nouns "John went to New York"

# Thematic replacement
scramblebench transform proper-nouns "Alice visited Paris" \
  --strategy thematic \
  --seed 42

# Phonetic similarity
scramblebench transform proper-nouns "Bob lives in Boston" \
  --strategy phonetic
```

**Strategies:**
- `random` - Random replacement from appropriate category
- `thematic` - Thematically consistent replacements
- `phonetic` - Phonetically similar replacements

### Synonym Replacement

```bash
# Replace 30% of words with synonyms
scramblebench transform synonyms "The big dog ran quickly"

# Custom replacement rate
scramblebench transform synonyms "This is a good example" \
  --replacement-rate 0.5 \
  --seed 123

# Include function words
scramblebench transform synonyms "The cat sat on the mat" \
  --replacement-rate 0.4 \
  --no-preserve-function-words
```

## Utility Commands

### Language Statistics

```bash
# Detailed statistics
scramblebench util stats mylang

# JSON output for processing
scramblebench util stats fantasy --output-format json
```

### Export Language Rules

```bash
# Export as JSON
scramblebench util export-rules mylang

# Export as CSV
scramblebench util export-rules fantasy \
  --output rules.csv \
  --format csv

# Export as YAML
scramblebench util export-rules mylang \
  --format yaml \
  --output custom_rules.yaml
```

### Validate Transformations

```bash
# Check if transformation is reversible
scramblebench util validate mylang "This is a test sentence"

# Validate with detailed output
scramblebench util validate fantasy "Complex example text" \
  --output-format json
```

## Output Formats

### Text (Default)
Human-readable tables and formatted output with colors.

### JSON
Machine-readable JSON format for automation:

```bash
scramblebench language list --output-format json
```

### YAML
YAML format for configuration and readability:

```bash
scramblebench language show mylang --output-format yaml
```

## Examples

### Complete Workflow Example

```bash
# 1. Generate a new language
scramblebench language generate scifi \
  --type synthetic \
  --complexity 6 \
  --vocab-size 1500 \
  --seed 42

# 2. Show language details
scramblebench language show scifi --show-rules --limit 20

# 3. Transform some text
scramblebench transform text "The astronaut explored the alien planet" scifi

# 4. Process a benchmark file
scramblebench batch transform benchmarks/qa.json scifi \
  --output scifi_qa.json

# 5. Get statistics
scramblebench util stats scifi

# 6. Export rules for analysis
scramblebench util export-rules scifi --format csv
```

### Batch Text Processing

```bash
# Extract vocabulary from multiple sources
scramblebench batch extract-vocab source1.json --output vocab1.json
scramblebench batch extract-vocab source2.json --output vocab2.json

# Generate language based on extracted vocabulary
scramblebench language generate domain_specific \
  --type phonetic \
  --complexity 5

# Transform multiple files
for file in benchmarks/*.json; do
  output="transformed/$(basename "$file" .json)_transformed.json"
  scramblebench batch transform "$file" domain_specific --output "$output"
done
```

### Proper Noun Anonymization

```bash
# Anonymize all proper nouns in text
scramblebench transform proper-nouns \
  "Dr. Smith from Boston Medical Center treated John Doe" \
  --strategy random \
  --seed 12345

# Output: "Dr. Henderson from Crystalport Medical Center treated Benjamin Quinn"
```

### Synonym Variation for Data Augmentation

```bash
# Create variations of training data
scramblebench transform synonyms \
  "The quick brown fox jumps over the lazy dog" \
  --replacement-rate 0.6 \
  --preserve-function-words

# Multiple variations with different seeds
for seed in {1..10}; do
  scramblebench transform synonyms \
    "Original training sentence" \
    --replacement-rate 0.4 \
    --seed $seed
done
```

## Advanced Usage

### Pipeline Processing

Combine multiple transformation steps:

```bash
# Step 1: Replace proper nouns
text=$(scramblebench transform proper-nouns \
  "John from New York likes Apple products" \
  --quiet --output-format json | jq -r '.transformed')

# Step 2: Replace synonyms
text=$(scramblebench transform synonyms "$text" \
  --replacement-rate 0.3 \
  --quiet --output-format json | jq -r '.transformed')

# Step 3: Apply language transformation
scramblebench transform text "$text" mylang
```

### Custom Language Creation Workflow

```bash
# 1. Extract domain-specific vocabulary
scramblebench batch extract-vocab domain_data.json \
  --min-freq 5 \
  --max-words 3000 \
  --output domain_vocab.json

# 2. Generate language with high complexity
scramblebench language generate domain_lang \
  --type synthetic \
  --complexity 8 \
  --vocab-size 3000

# 3. Test and validate
scramblebench util validate domain_lang \
  "Sample text from the domain"

# 4. Process entire dataset
scramblebench batch transform domain_data.json domain_lang \
  --output transformed_domain_data.json
```

## Configuration and Data Directories

The CLI uses the following directory structure:

```
data/
├── languages/          # Generated language files (.json)
├── benchmarks/         # Source benchmark files
├── results/           # Transformation results and exports
└── vocabularies/      # Extracted vocabularies
```

Use `--data-dir` to specify a different base directory:

```bash
scramblebench --data-dir /path/to/custom/data language list
```

## Error Handling and Debugging

### Verbose Mode
Use `-v` or `--verbose` for detailed operation logs:

```bash
scramblebench -v language generate debug_lang --type substitution
```

### Quiet Mode
Use `-q` or `--quiet` for minimal output:

```bash
scramblebench -q batch transform large_file.json mylang
```

### JSON Output for Automation
Use `--output-format json` for machine-readable output:

```bash
result=$(scramblebench --output-format json language show mylang)
echo "$result" | jq '.vocab_size'
```

## Performance Considerations

### Large File Processing
For large benchmark files, use appropriate batch sizes:

```bash
scramblebench batch transform huge_dataset.json mylang \
  --batch-size 500
```

### Memory Usage
Monitor memory usage when processing large vocabularies:

```bash
scramblebench batch extract-vocab large_corpus.json \
  --max-words 50000 \
  --min-freq 10
```

### Reproducibility
Always use seeds for reproducible results:

```bash
scramblebench language generate reproducible \
  --type phonetic \
  --seed 42

scramblebench transform synonyms "test text" \
  --seed 42 \
  --replacement-rate 0.5
```

This ensures identical results across runs and environments.