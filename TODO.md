# ScrambleBench Refinement — `todo.md`

**TL;DR:** Execute a comprehensive upgrade of the ScrambleBench toolkit to unify its architecture, bolster reliability via CI and testing, and enhance the research workflow with an ORM and interactive dashboard.

## Invariants (do not change)

*   **Deterministic Evaluation:** All evaluation runs must be fully reproducible. `temperature=0`, fixed seeds, and version-pinned environments are non-negotiable.
*   **Provider Isolation:** The provider used for generating paraphrases must *never* be used for evaluating model performance within the same experiment. This is a critical guardrail against data contamination.
*   **Single CLI Entry Point:** All core functionality must be accessible through the `scramblebench` command. Scripts are for automation, not primary interaction.
*   **Schema as Truth:** The `language_dependency_atlas.sql` schema and corresponding ORM models are the canonical representation of the research data structure. All data operations must respect their integrity.

## Assumptions & Scope

*   **Assumption:** The existing database schema (`language_dependency_atlas.sql`) and ORM models (`core/models.py`) are a robust and correct foundation for the research goals.
*   **Assumption:** The core research questions (discovering scaling laws, separating contamination from brittleness) remain the primary focus.
*   **Scope:** This plan covers enhancing the *toolkit* itself—improving code quality, research capabilities, and user experience. It does not define new scientific experiments but builds the instrument to conduct them.
*   **Placeholder:** `{{RUN_ID}}` refers to a specific, unique identifier for an evaluation run.

## Objectives

1.  **Solidify Codebase:** Achieve >90% test coverage on core modules (`core`, `analysis`, `transforms`, `db`) and establish a mandatory green CI pipeline that includes static analysis.
2.  **Enhance Research Engine:** Fully implement the SQLAlchemy ORM layer for all database interactions and add at least one new advanced statistical model (e.g., Bayesian hierarchical model) to the analysis suite.
3.  **Improve User Experience:** Fully implement the unified CLI design, deprecating standalone scripts, and create a functional Streamlit-based dashboard for interactive exploration of a completed run's results.

## Risks & Mitigations

*   **Risk:** The ORM refactor is complex and could introduce bugs or performance regressions. → **Mitigation:** Implement the ORM with a comprehensive integration test suite that validates key queries against both raw SQL and the ORM, ensuring identical results and performance within a 10% margin.
*   **Risk:** Advanced statistical models may be difficult for users to interpret correctly. → **Mitigation:** For each new model, create a dedicated Jupyter notebook tutorial using mock data that explains its purpose, assumptions, and how to interpret the output.
*   **Risk:** A unified CI pipeline becomes a bottleneck for development. → **Mitigation:** Structure the CI workflow with parallel jobs and smart caching to keep average pipeline duration under 15 minutes. Use `pytest-xdist` for parallel test execution.

## Method Outline (idea → mechanism → trade-offs → go/no-go)

### Workstream A — Architectural Unification

*   **Idea:** Create a single, canonical workflow for configuration, data access, and experiment execution.
*   **Mechanism:** Fully transition to the SQLAlchemy ORM for all database operations. Consolidate all execution logic from `core/runner.py` and `scripts/*.py` into the `experiment_tracking` system. Complete the migration of all functionality from standalone scripts into the `scramblebench` CLI.
*   **Trade-offs:** Initial development velocity will slow down to perform the refactor and write corresponding tests. This is an upfront investment in long-term stability and maintainability.
*   **Go/No-Go Gate:** The `smoke-test` can be run end-to-end using the unified ORM and `experiment_tracking` engine, producing results identical to the previous implementation.

### Workstream B — Research Workflow Upgrade

*   **Idea:** Streamline the end-to-end process from running an experiment to analyzing and visualizing its results.
*   **Mechanism:** Create an interactive Streamlit dashboard that takes a `{{RUN_ID}}` and visualizes key results by querying the database via the ORM. Tightly integrate the analysis pipeline so `scramblebench analyze` commands operate directly on a `{{RUN_ID}}`.
*   **Trade-offs:** Adds Streamlit as a new dependency. The dashboard will be read-only and focused on analysis, not experiment execution.
*   **Go/No-Go Gate:** A user can successfully initialize a project, run a smoke test, and view the results on the dashboard using only `scramblebench` CLI commands and `streamlit run`.

### Workstream C — Developer Experience & Reliability

*   **Idea:** Make the project robust, maintainable, and lower the barrier for new contributions.
*   **Mechanism:** Integrate `black`, `ruff`, and `mypy` into a single, mandatory CI workflow. Expand the `pytest` suite to cover the new ORM layer, the unified execution engine, and all CLI commands. Set up Sphinx for auto-generated API documentation deployed via GitHub Pages.
*   **Trade-offs:** A stricter, CI-gated development process may feel more rigid initially but will prevent technical debt and ensure code quality.
*   **Go/No-Go Gate:** The full CI pipeline passes consistently on the `main` branch, test coverage exceeds 90% for core modules, and API documentation is successfully deployed and accessible.

## Run Matrix

| ID | Method/Variant | Budget | Inputs | Expected Gain | Promote if… |
| -- | -------------- | ------ | ------ | ------------- | ----------- |
| V1 | Architectural Unification | Dev Time | Current codebase | Reduced complexity, improved maintainability | Smoke test passes with identical results; performance degradation ≤15%. |
| V2 | Research Workflow Upgrade | Dev Time | V1 codebase | Streamlined analysis, interactive exploration | Dashboard successfully visualizes a completed run from the database. |
| V3 | DevEx & Reliability | Dev Time | V1+V2 codebase | Fewer bugs, better contribution experience | CI pipeline green, test coverage >90%, docs deployed. |

## Implementation Notes

*   **APIs/Attach points:** All model interactions must go through the `BaseModelAdapter` interface. All database interactions will go through a new `DatabaseManager` class wrapping the SQLAlchemy session.
*   **Precision/Quantization:** All statistical computations to be done in `fp64`. Model inference can use `bf16`/`fp16` where supported.
*   **Caching/State:** Paraphrase cache remains immutable. ORM results can leverage SQLAlchemy's identity map for session-level caching. The dashboard will use Streamlit's caching for expensive queries.
*   **Telemetry:** Log structured events (e.g., `experiment_start`, `model_evaluation_complete`, `analysis_failed`) with context (`run_id`, `model_id`, etc.) using the existing structured logger.
*   **Repro:** `poetry.lock` file must be committed. Every run must be associated with a git SHA. The environment snapshot in the `experiments` table must be comprehensive.

## Acceptance Gates

*   **Objective 1 (Solidify):** The CI workflow defined in `building` and `running` workflows passes for all PRs to `main`. `pytest --cov` reports >90% coverage for `core`, `db`, `analysis`, and `transforms`.
*   **Objective 2 (Enhance):** All data access in the primary evaluation and analysis paths uses the SQLAlchemy ORM. A `scramblebench analyze fit --model bayesian` command is implemented and tested.
*   **Objective 3 (Usability):** `run_smoke_test.py` and `run_scaling_survey.py` are removed, and their functionality is available via `scramblebench smoke-test` and `scramblebench survey run`. A user can run `streamlit run src/scramblebench/dashboard/app.py` and interactively plot results from a completed run.

## “Make-sure-you” Checklist

*   All new database interactions *must* use the ORM layer. No new raw SQL queries in application code.
*   CI must pass all quality, test, and build stages before any PR can be merged to `main`.
*   Any change to the database schema must be accompanied by an Alembic migration script.
*   All new user-facing features must have a corresponding section in the Sphinx documentation.

## File/Layout Plan

```
scramblebench/
  src/scramblebench/
    analysis/       # Statistical analysis, visualization, export
    core/           # Core logic: config, adapters (runner deprecated)
    db/             # NEW: SQLAlchemy models, session manager, repository
    evaluation/     # High-level evaluation logic (runner deprecated)
    experiment_tracking/ # PRIMARY execution engine
    llm/            # Low-level LLM client implementations
    transforms/     # Transformation strategies
    utils/          # General utilities
    dashboard/      # NEW: Streamlit dashboard application
    cli.py          # Main CLI entry point
  configs/          # Example YAML configs
  db/               # SQLite/DuckDB database file
  docs/             # Sphinx documentation
  tests/            # Pytest suite
  alembic/          # NEW: Alembic migration scripts
  pyproject.toml
  poetry.lock
  README.md
```

## Workflows (required)

```xml
<workflows project="scramblebench_refinement" version="2.0">

  <!-- =============================== -->
  <!-- BUILDING: env, assets, guards   -->
  <!-- =============================== -->
  <workflow name="building">
    <env id="B0">
      <desc>Set up development environment and pin versions</desc>
      <commands>
        <cmd>poetry install --with dev,analysis,docs</cmd>
        <cmd>poetry lock --no-update</cmd>
        <cmd>pre-commit install</cmd>
      </commands>
      <make_sure>
        <item>Ollama server is running for local tests.</item>
        <item>`poetry.lock` is committed and up-to-date.</item>
      </make_sure>
    </env>
    <assets id="B1">
      <desc>Initialize and migrate the database</desc>
      <commands>
        <cmd>scramblebench db init  # Applies Alembic migrations</cmd>
        <cmd>python scripts/generate_mock_data.py</cmd>
      </commands>
      <make_sure>
        <item>Database schema matches the ORM models.</item>
        <item>Mock data for smoke tests is present.</item>
      </make_sure>
    </assets>
    <guards id="B2">
      <desc>Run static analysis and quality guards</desc>
      <commands>
        <cmd>poetry run black . --check</cmd>
        <cmd>poetry run ruff check .</cmd>
        <cmd>poetry run mypy src/</cmd>
      </commands>
      <make_sure>
        <item>All static analysis checks pass before committing.</item>
      </make_sure>
    </guards>
  </workflow>

  <!-- =============================== -->
  <!-- RUNNING: core & variants        -->
  <!-- =============================== -->
  <workflow name="running">
    <baseline id="R0">
      <desc>Run full test suite as the baseline validation</desc>
      <commands>
        <cmd>poetry run pytest tests/ --cov=src/scramblebench</cmd>
      </commands>
      <make_sure>
        <item>Test coverage meets or exceeds the target (>90%).</item>
      </make_sure>
    </baseline>
    <variants id="R1">
      <desc>Run smoke test to validate the unified CLI and ORM stack</desc>
      <commands>
        <cmd>scramblebench smoke-test --config configs/smoke.yaml</cmd>
      </commands>
      <make_sure>
        <item>Smoke test completes successfully and populates the database via the ORM.</item>
      </make_sure>
    </variants>
  </workflow>

  <!-- =============================== -->
  <!-- TRACKING: collect & compute     -->
  <!-- =============================== -->
  <workflow name="tracking">
    <harvest id="T1">
      <desc>Analyze a completed run and view results</desc>
      <commands>
        <cmd>scramblebench analyze summary --run-id {{RUN_ID}}</cmd>
        <cmd>scramblebench analyze fit --run-id {{RUN_ID}} --model all --export-latex</cmd>
        <cmd>streamlit run src/scramblebench/dashboard/app.py -- --run-id {{RUN_ID}}</cmd>
      </commands>
      <make_sure>
        <item>Analysis results and tables are saved correctly.</item>
        <item>Dashboard successfully loads and visualizes the run data.</item>
      </make_sure>
    </harvest>
  </workflow>

  <!-- =============================== -->
  <!-- EVALUATING: promotion rules     -->
  <!-- =============================== -->
  <workflow name="evaluating">
    <promote id="E1">
      <desc>Validate enhancements against acceptance gates</desc>
      <commands>
        <cmd>python scripts/validate_orm_parity.py --run-id-legacy {{LEGACY_RUN_ID}} --run-id-orm {{ORM_RUN_ID}}</cmd>
        <cmd>python scripts/check_coverage.py --threshold 90</cmd>
        <cmd>scramblebench docs build --deploy</cmd>
      </commands>
      <make_sure>
        <item>ORM-based runs produce identical results to legacy runs.</item>
        <item>Code coverage meets the >90% target.</item>
        <item>Documentation is successfully deployed to GitHub Pages.</item>
      </make_sure>
    </promote>
  </workflow>

  <!-- =============================== -->
  <!-- REFINEMENT: next iteration      -->
  <!-- =============================== -->
  <workflow name="refinement">
    <next id="N1">
      <desc>Plan next feature based on validated foundation</desc>
      <commands>
        <cmd># (Manual) Review project board and prioritize next workstream (e.g., new models, new transforms).</cmd>
        <cmd># (Manual) Update this `todo.md` with the next set of objectives.</cmd>
      </commands>
      <make_sure>
        <item>The project's `todo.md` is updated to reflect the next set of objectives.</item>
      </make_sure>
    </next>
  </workflow>

</workflows>
```

## Minimal Pseudocode (optional)

```python
# Illustrates the intended shift from raw SQL/DuckDB to ORM-based data access.

# OLD WAY (example from core/database.py)
def get_run_status_duckdb(db_conn, run_id: str) -> dict:
    query = "SELECT status, completed_evaluations FROM runs WHERE run_id = ?"
    result = db_conn.execute(query, (run_id,)).fetchone()
    return {'status': result[0], 'progress': result[1]}

# NEW WAY (with SQLAlchemy ORM via Repository)
from scramblebench.db.models import Run
from scramblebench.db.repository import RunRepository

def get_run_status_orm(run_repo: RunRepository, run_id: str) -> dict:
    """Type-safe, more maintainable query using the ORM."""
    run = run_repo.get_by_id(run_id)
    if not run:
        return {}
    return {'status': run.status, 'progress': run.completed_evaluations}
```

## Next Actions (strict order)

1.  **CI Solidification:** Merge `ci.yml` and `docs.yml` into a single, comprehensive workflow. Add `black`, `ruff`, and `mypy` checks to the CI pipeline and make it a mandatory check for all PRs to `main`.
2.  **Configuration Unification:** Systematically refactor the entire codebase to import and use `ScrambleBenchConfig` from `src/scramblebench/core/unified_config.py`. Remove the deprecated config files (`core/config.py`, `evaluation/config.py`, `utils/config.py`).
3.  **ORM Scaffolding:** Ensure `alembic/` is fully functional and the ORM models in `src/scramblebench/core/models.py` are the single source of truth for the database schema.
4.  **Test Suite Expansion:** Begin writing unit and integration tests for the `core/repository.py` classes and the `experiment_tracking` execution engine.
5.  **CLI Refactor:** Migrate the logic from `scripts/run_smoke_test.py` into a `scramblebench smoke-test` command. Deprecate and remove the original script.
6.  **Dashboard Scaffolding:** Create the initial Streamlit application file in `src/scramblebench/dashboard/app.py` with basic database connection and placeholder plots.