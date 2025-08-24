"""
Command Line Interface for Experiment Tracking

Comprehensive CLI for managing experiments, monitoring progress, 
and accessing tracking system functionality.
"""

import asyncio
import click
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

from .core import ExperimentTracker, ExperimentStatus
from .database import DatabaseManager
from .monitor import ExperimentMonitor
from .statistics import StatisticalAnalyzer
from .reproducibility import ReproducibilityValidator
from ..core.unified_config import ScrambleBenchConfig


@click.group()
@click.option('--database-url', envvar='DATABASE_URL', 
              help='PostgreSQL database URL')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
@click.pass_context
def cli(ctx, database_url: str, verbose: bool):
    """ScrambleBench Experiment Tracking CLI"""
    ctx.ensure_object(dict)
    
    if not database_url:
        click.echo("Error: DATABASE_URL is required", err=True)
        sys.exit(1)
    
    ctx.obj['database_url'] = database_url
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('name')
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--description', '-d', required=True, help='Experiment description')
@click.option('--research-question', '-q', required=True, help='Research question')
@click.option('--hypothesis', '-h', help='Hypothesis (optional)')
@click.option('--researcher', '-r', required=True, help='Researcher name')
@click.option('--institution', '-i', help='Research institution')
@click.option('--priority', '-p', type=int, default=1, help='Queue priority')
@click.option('--tags', help='Comma-separated tags')
@click.pass_context
def create(ctx, name: str, config_path: str, description: str, research_question: str,
          hypothesis: Optional[str], researcher: str, institution: Optional[str],
          priority: int, tags: Optional[str]):
    """Create a new experiment"""
    
    async def _create_experiment():
        tracker = ExperimentTracker(ctx.obj['database_url'])
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
        
        experiment_id = await tracker.create_experiment(
            name=name,
            config=Path(config_path),
            description=description,
            research_question=research_question,
            researcher_name=researcher,
            hypothesis=hypothesis,
            institution=institution,
            priority=priority,
            tags=tag_list
        )
        
        click.echo(f"Created experiment: {experiment_id}")
        return experiment_id
    
    experiment_id = asyncio.run(_create_experiment())
    
    # Auto-queue if requested
    if click.confirm("Queue experiment for execution?", default=True):
        asyncio.run(_queue_experiment(ctx, experiment_id, priority))


async def _queue_experiment(ctx, experiment_id: str, priority: int):
    """Queue an experiment for execution"""
    tracker = ExperimentTracker(ctx.obj['database_url'])
    await tracker.queue_experiment(experiment_id, priority=priority)
    click.echo(f"Queued experiment {experiment_id} with priority {priority}")


@cli.command()
@click.argument('experiment_id')
@click.option('--priority', '-p', type=int, default=1, help='Queue priority')
@click.pass_context
def queue(ctx, experiment_id: str, priority: int):
    """Queue an experiment for execution"""
    asyncio.run(_queue_experiment(ctx, experiment_id, priority))


@cli.command()
@click.option('--continuous', '-c', is_flag=True, help='Run continuously')
@click.option('--max-concurrent', '-m', type=int, default=3, help='Max concurrent experiments')
@click.option('--check-interval', type=int, default=30, help='Check interval in seconds')
@click.pass_context
def run(ctx, continuous: bool, max_concurrent: int, check_interval: int):
    """Start the experiment runner"""
    
    async def _run_experiments():
        tracker = ExperimentTracker(
            ctx.obj['database_url'],
            max_concurrent_experiments=max_concurrent
        )
        
        click.echo(f"Starting experiment runner (max concurrent: {max_concurrent})")
        
        if continuous:
            click.echo("Running continuously... Press Ctrl+C to stop")
        
        try:
            await tracker.run_experiments(
                continuous=continuous,
                check_interval=check_interval
            )
        except KeyboardInterrupt:
            click.echo("Shutting down experiment runner...")
            tracker.shutdown()
    
    asyncio.run(_run_experiments())


@cli.command()
@click.argument('experiment_id')
@click.pass_context
def status(ctx, experiment_id: str):
    """Get experiment status"""
    
    async def _get_status():
        tracker = ExperimentTracker(ctx.obj['database_url'])
        status_data = await tracker.get_experiment_status(experiment_id)
        
        if 'error' in status_data:
            click.echo(f"Error: {status_data['error']}", err=True)
            return
        
        # Format status output
        click.echo(f"\nExperiment: {status_data['name']}")
        click.echo(f"ID: {status_data['experiment_id']}")
        click.echo(f"Status: {status_data['status']}")
        
        if status_data['progress'] > 0:
            click.echo(f"Progress: {status_data['progress']*100:.1f}%")
            click.echo(f"Current Stage: {status_data['current_stage']}")
        
        if status_data.get('created_at'):
            click.echo(f"Created: {status_data['created_at']}")
        
        if status_data.get('started_at'):
            click.echo(f"Started: {status_data['started_at']}")
        
        if status_data.get('completed_at'):
            click.echo(f"Completed: {status_data['completed_at']}")
        
        if status_data.get('total_cost', 0) > 0:
            click.echo(f"Cost: ${status_data['total_cost']:.4f}")
        
        if status_data.get('compute_hours', 0) > 0:
            click.echo(f"Compute Hours: {status_data['compute_hours']:.2f}")
    
    asyncio.run(_get_status())


@cli.command()
@click.option('--status', '-s', type=click.Choice(['planned', 'queued', 'running', 'completed', 'failed', 'cancelled']),
              help='Filter by status')
@click.option('--researcher', '-r', help='Filter by researcher')
@click.option('--limit', '-l', type=int, default=20, help='Maximum results to show')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.pass_context
def list(ctx, status: Optional[str], researcher: Optional[str], limit: int, output_format: str):
    """List experiments"""
    
    async def _list_experiments():
        tracker = ExperimentTracker(ctx.obj['database_url'])
        
        status_filter = ExperimentStatus(status) if status else None
        experiments = await tracker.list_experiments(
            status=status_filter,
            researcher=researcher,
            limit=limit
        )
        
        if not experiments:
            click.echo("No experiments found")
            return
        
        if output_format == 'json':
            click.echo(json.dumps(experiments, indent=2))
        else:
            # Table format
            click.echo(f"\nFound {len(experiments)} experiments:\n")
            
            # Headers
            click.echo(f"{'ID':<8} {'Name':<20} {'Status':<10} {'Researcher':<15} {'Created':<19}")
            click.echo("-" * 80)
            
            # Rows
            for exp in experiments:
                exp_id = exp['experiment_id'][:8]
                name = exp['experiment_name'][:20]
                status = exp['status'][:10]
                researcher = exp['researcher_name'][:15]
                created = exp['created_at'][:19] if exp['created_at'] else 'N/A'
                
                click.echo(f"{exp_id:<8} {name:<20} {status:<10} {researcher:<15} {created:<19}")
    
    asyncio.run(_list_experiments())


@cli.command()
@click.argument('experiment_id')
@click.pass_context
def cancel(ctx, experiment_id: str):
    """Cancel a running or queued experiment"""
    
    async def _cancel_experiment():
        tracker = ExperimentTracker(ctx.obj['database_url'])
        await tracker.cancel_experiment(experiment_id)
        click.echo(f"Cancelled experiment {experiment_id}")
    
    if click.confirm(f"Cancel experiment {experiment_id}?"):
        asyncio.run(_cancel_experiment())


@cli.command()
@click.argument('experiment_id')
@click.option('--output-dir', '-o', type=click.Path(), 
              help='Output directory (default: ./analysis_<experiment_id>)')
@click.option('--include-plots', is_flag=True, help='Generate plots')
@click.option('--include-raw-data', is_flag=True, help='Include raw data export')
@click.pass_context
def analyze(ctx, experiment_id: str, output_dir: Optional[str], 
           include_plots: bool, include_raw_data: bool):
    """Run statistical analysis on experiment results"""
    
    async def _run_analysis():
        db_manager = DatabaseManager(ctx.obj['database_url'])
        analyzer = StatisticalAnalyzer(db_manager)
        
        # Set output directory
        if not output_dir:
            output_dir_path = Path(f"analysis_{experiment_id}")
        else:
            output_dir_path = Path(output_dir)
        
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Running statistical analysis for experiment {experiment_id}")
        
        # Get experiment data
        experiment_data = await db_manager.get_experiment_data(experiment_id)
        
        if not experiment_data:
            click.echo("No data found for experiment", err=True)
            return
        
        # Generate summary statistics
        summary = await db_manager.get_performance_summary(experiment_id)
        
        summary_file = output_dir_path / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        click.echo(f"Analysis saved to {output_dir_path}")
        
        # Display key results
        click.echo(f"\nSummary:")
        click.echo(f"  Total responses: {summary.get('total_responses', 0)}")
        click.echo(f"  Overall accuracy: {summary.get('accuracy', 0)*100:.2f}%")
        click.echo(f"  Average response time: {summary.get('avg_response_time_ms', 0):.1f}ms")
        click.echo(f"  Total cost: ${summary.get('total_cost', 0):.4f}")
        click.echo(f"  Models tested: {summary.get('models_tested', 0)}")
    
    asyncio.run(_run_analysis())


@cli.command()
@click.argument('experiment_id')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for replication package')
@click.option('--include-data', is_flag=True, help='Include data files')
@click.pass_context
def replicate(ctx, experiment_id: str, output_dir: Optional[str], include_data: bool):
    """Generate replication package for experiment"""
    
    async def _generate_package():
        db_manager = DatabaseManager(ctx.obj['database_url'])
        validator = ReproducibilityValidator()
        
        # Get experiment metadata
        metadata = await db_manager.get_experiment_metadata(experiment_id)
        if not metadata:
            click.echo(f"Experiment {experiment_id} not found", err=True)
            return
        
        # Set output directory
        if not output_dir:
            output_dir_path = Path(f"replication_{experiment_id}")
        else:
            output_dir_path = Path(output_dir)
        
        # Find configuration file
        config_path = Path(f"experiments/{experiment_id}/config.yaml")
        if not config_path.exists():
            click.echo("Configuration file not found", err=True)
            return
        
        click.echo(f"Generating replication package for {metadata.name}")
        
        package = await validator.generate_replication_package(
            experiment_id=experiment_id,
            experiment_name=metadata.name,
            config_path=config_path,
            output_dir=output_dir_path,
            include_data=include_data
        )
        
        click.echo(f"Replication package generated at {output_dir_path}")
        click.echo(f"Package includes {len(package.file_checksums)} files")
    
    asyncio.run(_generate_package())


@cli.command()
@click.pass_context
def monitor(ctx):
    """Start monitoring dashboard (web interface)"""
    
    async def _start_monitor():
        db_manager = DatabaseManager(ctx.obj['database_url'])
        monitor = ExperimentMonitor(db_manager)
        
        await monitor.start_monitoring()
        
        click.echo("Monitoring dashboard started at http://localhost:8080")
        click.echo("Press Ctrl+C to stop")
        
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await monitor.stop_monitoring()
            click.echo("Monitoring stopped")
    
    asyncio.run(_start_monitor())


@cli.command()
@click.pass_context
def queue_status(ctx):
    """Show experiment queue status"""
    
    async def _show_queue_status():
        tracker = ExperimentTracker(ctx.obj['database_url'])
        queue_summary = tracker.queue.get_queue_summary()
        
        click.echo("\nExperiment Queue Status:")
        click.echo(f"  Total experiments: {queue_summary['total_experiments']}")
        click.echo(f"  Currently running: {queue_summary['running']}")
        click.echo(f"  Ready in queue: {queue_summary['ready_in_queue']}")
        click.echo(f"  Completed: {queue_summary['completed']}")
        click.echo(f"  Failed: {queue_summary['failed']}")
        
        if 'next_experiments' in queue_summary:
            click.echo(f"\nNext experiments to run:")
            for exp in queue_summary['next_experiments']:
                click.echo(f"  - {exp['experiment_id'][:8]}: Priority {exp['priority']}")
        
        # Resource usage
        usage = queue_summary['current_resource_usage']
        limits = queue_summary['resource_limits']
        
        click.echo(f"\nResource Usage:")
        for resource, current in usage.items():
            limit = limits.get(resource, 0)
            if limit > 0:
                percentage = (current / limit) * 100
                click.echo(f"  {resource}: {current:.1f}/{limit:.1f} ({percentage:.1f}%)")
    
    asyncio.run(_show_queue_status())


@cli.command()
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Configuration file to validate')
@click.pass_context
def validate_config(ctx, config_file: str):
    """Validate experiment configuration file"""
    
    try:
        config = EvaluationConfig.load_from_file(config_file)
        click.echo(f"✓ Configuration file is valid")
        
        # Show summary
        click.echo(f"  Experiment: {config.experiment_name}")
        click.echo(f"  Models: {len(config.models)}")
        click.echo(f"  Benchmarks: {len(config.benchmark_paths)}")
        
        if config.max_samples:
            click.echo(f"  Max samples: {config.max_samples}")
        
        # Estimate resources
        tracker = ExperimentTracker(ctx.obj['database_url'])
        estimates = asyncio.run(tracker._estimate_resource_requirements(config))
        
        click.echo(f"\nEstimated Resources:")
        click.echo(f"  API calls: {estimates['api_calls']:,}")
        click.echo(f"  Duration: {estimates['estimated_duration_hours']:.1f} hours")
        if estimates['estimated_cost'] > 0:
            click.echo(f"  Cost: ${estimates['estimated_cost']:.2f}")
        
    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)
        sys.exit(1)


@cli.command() 
@click.option('--environment-file', '-e', type=click.Path(exists=True),
              help='Environment snapshot file to validate against')
@click.option('--tolerance', type=click.Choice(['strict', 'moderate', 'loose']), 
              default='moderate', help='Validation tolerance')
@click.pass_context
def validate_environment(ctx, environment_file: str, tolerance: str):
    """Validate current environment for reproducibility"""
    
    async def _validate_environment():
        validator = ReproducibilityValidator()
        
        if environment_file:
            # Load original environment
            with open(environment_file, 'r') as f:
                original_env_data = json.load(f)
            
            from .reproducibility import EnvironmentSnapshot
            original_snapshot = EnvironmentSnapshot.from_dict(original_env_data)
            
            # Validate against current environment
            validation = await validator.validate_reproducibility(
                original_snapshot, tolerance
            )
            
            click.echo(f"Environment Validation ({tolerance} tolerance):")
            
            if validation['is_reproducible']:
                click.echo(f"✓ Environment is reproducible (confidence: {validation['confidence']})")
            else:
                click.echo(f"✗ Environment validation failed")
            
            if validation['issues']:
                click.echo(f"\nIssues:")
                for issue in validation['issues']:
                    click.echo(f"  - {issue}")
            
            if validation['warnings']:
                click.echo(f"\nWarnings:")
                for warning in validation['warnings']:
                    click.echo(f"  - {warning}")
        
        else:
            # Just capture current environment
            env_data = await validator.capture_environment()
            
            click.echo("Current Environment Snapshot:")
            click.echo(f"  Python: {env_data['python_version']}")
            click.echo(f"  Platform: {env_data['platform_system']} {env_data['platform_release']}")
            click.echo(f"  Packages: {len(env_data['installed_packages'])}")
            click.echo(f"  Git commit: {env_data['git_commit_hash']}")
            
            # Save snapshot
            output_file = f"environment_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(env_data, f, indent=2)
            
            click.echo(f"\nEnvironment snapshot saved to {output_file}")
    
    asyncio.run(_validate_environment())


if __name__ == '__main__':
    cli()