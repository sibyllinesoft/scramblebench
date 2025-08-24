# ⚠️ DEPRECATED: run_smoke_test.py

**This standalone script has been deprecated and removed.**

## Migration to Unified CLI

All functionality from `scripts/run_smoke_test.py` has been migrated to the unified CLI command:

```bash
# Old way (REMOVED)
python scripts/run_smoke_test.py --config myconfig.yaml --max-cost 5.0

# New way (USE THIS)
scramblebench smoke-test --config myconfig.yaml --max-cost 5.0
```

## Complete Feature Mapping

All command-line arguments are preserved in the CLI version:

| Standalone Script | CLI Command | Description |
|-------------------|-------------|-------------|
| `--config` | `--config` | Base configuration file |
| `--output-dir` | `--output-dir` | Output directory for results |
| `--max-cost` | `--max-cost` | Maximum cost limit in USD |
| `--timeout` | `--timeout` | Timeout in minutes |
| `--items` | `--items` | Maximum items per dataset |
| `--models` | `--models` | Maximum models to test |
| `--scramble-levels` | `--scramble-levels` | Scramble levels (can specify multiple) |
| `--force` | `--force` | Force execution despite budget |
| `--ci` | `--ci` | CI mode with proper exit codes |
| `--verbose` | *(global)* | Use global `--verbose` flag |
| `--quiet` | *(global)* | Use global `--quiet` flag |
| `--export-ci-config` | `--export-ci-config` | Export GitHub Actions config |

## Example Migration

**Old Command:**
```bash
python scripts/run_smoke_test.py \
  --config configs/smoke.yaml \
  --max-cost 3.0 \
  --items 15 \
  --models 3 \
  --scramble-levels 0.1 0.3 0.5 \
  --ci \
  --verbose
```

**New Command:**
```bash
scramblebench --verbose smoke-test \
  --config configs/smoke.yaml \
  --max-cost 3.0 \
  --items 15 \
  --models 3 \
  --scramble-levels 0.1 \
  --scramble-levels 0.3 \
  --scramble-levels 0.5 \
  --ci
```

## Additional Benefits

The unified CLI provides:

- ✅ **Consistent interface** with all other ScrambleBench commands
- ✅ **Better error handling** with global verbose/quiet modes
- ✅ **Unified output formatting** (text/JSON)
- ✅ **Integration with global configuration** and context management
- ✅ **Enhanced CI integration** with proper exit codes
- ✅ **Future-proof architecture** following TODO.md requirements

## Migration Date

- **Deprecated:** 2025-01-23
- **Removed:** 2025-01-23
- **Replacement:** `scramblebench smoke-test`

For any issues with the migration, please refer to the unified CLI documentation or raise an issue.