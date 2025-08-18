#!/usr/bin/env python3
"""
Release automation script for ScrambleBench.

This script automates the release process including:
- Version bumping
- Changelog generation
- Git tagging
- Building and publishing
- Documentation updates
"""

import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scramblebench.core.logging import configure_logging, get_logger


class ReleaseManager:
    """Manages the release process for ScrambleBench."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = get_logger(__name__)
        self.project_root = Path(__file__).parent.parent
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.changelog_path = self.project_root / "CHANGELOG.md"
        
        if dry_run:
            self.logger.info("🔍 Running in DRY RUN mode - no changes will be made")
    
    def run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        self.logger.debug(f"Running command: {' '.join(cmd)}")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=True,
                cwd=self.project_root
            )
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(cmd)}")
            self.logger.error(f"Exit code: {e.returncode}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            raise
    
    def get_current_version(self) -> str:
        """Get the current version from pyproject.toml."""
        with open(self.pyproject_path, 'r') as f:
            content = f.read()
        
        version_match = re.search(r'version = "([^"]+)"', content)
        if not version_match:
            raise ValueError("Could not find version in pyproject.toml")
        
        return version_match.group(1)
    
    def bump_version(self, bump_type: str) -> str:
        """Bump version using poetry."""
        self.logger.info(f"Bumping version: {bump_type}")
        
        # Use poetry to bump version
        self.run_command(["poetry", "version", bump_type])
        
        # Get the new version
        new_version = self.get_current_version()
        self.logger.info(f"New version: {new_version}")
        
        return new_version
    
    def get_git_commits_since_tag(self, tag: Optional[str] = None) -> List[Dict[str, str]]:
        """Get git commits since the last tag."""
        if tag is None:
            # Get the latest tag
            try:
                result = self.run_command(["git", "describe", "--tags", "--abbrev=0"])
                tag = result.stdout.strip()
            except subprocess.CalledProcessError:
                # No tags exist, get all commits
                tag = ""
        
        # Get commits since tag
        cmd = ["git", "log", "--pretty=format:%H|%s|%an|%ad", "--date=short"]
        if tag:
            cmd.append(f"{tag}..HEAD")
        
        result = self.run_command(cmd)
        
        commits = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('|', 3)
                if len(parts) == 4:
                    commits.append({
                        'hash': parts[0],
                        'message': parts[1],
                        'author': parts[2],
                        'date': parts[3]
                    })
        
        return commits
    
    def categorize_commits(self, commits: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Categorize commits by type."""
        categories = {
            'Features': [],
            'Bug Fixes': [],
            'Performance': [],
            'Documentation': [],
            'Tests': [],
            'Chores': [],
            'Breaking Changes': [],
            'Other': []
        }
        
        for commit in commits:
            message = commit['message'].lower()
            
            if any(word in message for word in ['feat:', 'feature:', 'add', 'new']):
                categories['Features'].append(commit)
            elif any(word in message for word in ['fix:', 'bug:', 'patch:', 'hotfix:']):
                categories['Bug Fixes'].append(commit)
            elif any(word in message for word in ['perf:', 'performance', 'optimize', 'speed']):
                categories['Performance'].append(commit)
            elif any(word in message for word in ['docs:', 'doc:', 'documentation']):
                categories['Documentation'].append(commit)
            elif any(word in message for word in ['test:', 'tests:', 'testing']):
                categories['Tests'].append(commit)
            elif any(word in message for word in ['chore:', 'style:', 'refactor:', 'cleanup']):
                categories['Chores'].append(commit)
            elif any(word in message for word in ['breaking', 'major:', 'breaking change']):
                categories['Breaking Changes'].append(commit)
            else:
                categories['Other'].append(commit)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def generate_changelog_entry(self, version: str, commits: List[Dict[str, str]]) -> str:
        """Generate changelog entry for the version."""
        date = datetime.now().strftime("%Y-%m-%d")
        
        lines = [
            f"## [{version}] - {date}",
            ""
        ]
        
        categorized = self.categorize_commits(commits)
        
        for category, category_commits in categorized.items():
            if category_commits:
                lines.append(f"### {category}")
                lines.append("")
                
                for commit in category_commits:
                    # Clean up commit message
                    message = commit['message']
                    # Remove conventional commit prefixes
                    message = re.sub(r'^(feat|fix|docs|style|refactor|test|chore|perf)(\([^)]+\))?: ', '', message)
                    message = message[0].upper() + message[1:] if message else message
                    
                    lines.append(f"- {message} ({commit['hash'][:8]})")
                
                lines.append("")
        
        return '\n'.join(lines)
    
    def update_changelog(self, version: str, commits: List[Dict[str, str]]) -> None:
        """Update CHANGELOG.md with new version."""
        self.logger.info("Updating CHANGELOG.md")
        
        new_entry = self.generate_changelog_entry(version, commits)
        
        if self.changelog_path.exists():
            with open(self.changelog_path, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        
        # Insert new entry after the header
        lines = existing_content.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('##') or (i > 5 and line.strip() == ''):
                header_end = i
                break
        
        new_lines = lines[:header_end] + [''] + new_entry.split('\n') + [''] + lines[header_end:]
        new_content = '\n'.join(new_lines)
        
        if not self.dry_run:
            with open(self.changelog_path, 'w') as f:
                f.write(new_content)
        
        self.logger.info(f"Updated CHANGELOG.md with {len(commits)} commits")
    
    def run_tests(self) -> None:
        """Run the test suite."""
        self.logger.info("Running test suite...")
        
        try:
            self.run_command(["poetry", "run", "pytest", "-v"], capture_output=False)
            self.logger.info("[CHECK] All tests passed")
        except subprocess.CalledProcessError:
            self.logger.error("[X] Tests failed")
            if not self.dry_run:
                raise
    
    def run_quality_checks(self) -> None:
        """Run code quality checks."""
        self.logger.info("Running quality checks...")
        
        checks = [
            (["poetry", "run", "black", "--check", "."], "Code formatting"),
            (["poetry", "run", "ruff", "check", "."], "Linting"),
            (["poetry", "run", "mypy", "src/"], "Type checking"),
        ]
        
        for cmd, description in checks:
            try:
                self.run_command(cmd)
                self.logger.info(f"[CHECK] {description} passed")
            except subprocess.CalledProcessError:
                self.logger.error(f"[X] {description} failed")
                if not self.dry_run:
                    raise
    
    def build_package(self) -> None:
        """Build the package."""
        self.logger.info("Building package...")
        
        # Clean previous builds
        dist_dir = self.project_root / "dist"
        if dist_dir.exists() and not self.dry_run:
            import shutil
            shutil.rmtree(dist_dir)
        
        self.run_command(["poetry", "build"])
        self.logger.info("[CHECK] Package built successfully")
    
    def create_git_tag(self, version: str) -> None:
        """Create and push git tag."""
        tag_name = f"v{version}"
        self.logger.info(f"Creating git tag: {tag_name}")
        
        # Create annotated tag
        self.run_command([
            "git", "tag", "-a", tag_name, 
            "-m", f"Release version {version}"
        ])
        
        # Push tag
        self.run_command(["git", "push", "origin", tag_name])
        self.logger.info(f"[CHECK] Tag {tag_name} created and pushed")
    
    def commit_release_changes(self, version: str) -> None:
        """Commit release-related changes."""
        self.logger.info("Committing release changes...")
        
        # Add changed files
        files_to_add = ["pyproject.toml"]
        if self.changelog_path.exists():
            files_to_add.append("CHANGELOG.md")
        
        for file in files_to_add:
            self.run_command(["git", "add", file])
        
        # Commit changes
        commit_message = f"chore: release version {version}\n\n🤖 Generated with ScrambleBench release automation"
        self.run_command(["git", "commit", "-m", commit_message])
        
        # Push changes
        self.run_command(["git", "push", "origin", "main"])
        self.logger.info("[CHECK] Release changes committed and pushed")
    
    def publish_to_pypi(self, test: bool = False) -> None:
        """Publish package to PyPI."""
        if test:
            self.logger.info("Publishing to Test PyPI...")
            self.run_command(["poetry", "publish", "--repository", "testpypi"])
        else:
            self.logger.info("Publishing to PyPI...")
            self.run_command(["poetry", "publish"])
        
        self.logger.info("[CHECK] Package published successfully")
    
    def create_github_release(self, version: str, commits: List[Dict[str, str]]) -> None:
        """Create GitHub release."""
        if self.dry_run:
            self.logger.info("[DRY RUN] Would create GitHub release")
            return
        
        self.logger.info("Creating GitHub release...")
        
        # Generate release notes
        release_notes = self.generate_changelog_entry(version, commits)
        
        try:
            # Use GitHub CLI if available
            self.run_command([
                "gh", "release", "create", f"v{version}",
                "--title", f"Release {version}",
                "--notes", release_notes,
                "dist/*"
            ])
            self.logger.info("[CHECK] GitHub release created")
        except subprocess.CalledProcessError:
            self.logger.warning("Could not create GitHub release (gh CLI not available)")
    
    def full_release(
        self,
        bump_type: str,
        skip_tests: bool = False,
        skip_quality: bool = False,
        test_pypi: bool = False
    ) -> str:
        """Perform a full release."""
        self.logger.info(f"[ROCKET] Starting release process (bump: {bump_type})")
        
        try:
            # Pre-release checks
            if not skip_quality:
                self.run_quality_checks()
            
            if not skip_tests:
                self.run_tests()
            
            # Get commits for changelog
            commits = self.get_git_commits_since_tag()
            self.logger.info(f"Found {len(commits)} commits since last release")
            
            # Bump version
            new_version = self.bump_version(bump_type)
            
            # Update changelog
            if commits:
                self.update_changelog(new_version, commits)
            
            # Build package
            self.build_package()
            
            # Commit and tag
            self.commit_release_changes(new_version)
            self.create_git_tag(new_version)
            
            # Publish
            self.publish_to_pypi(test=test_pypi)
            
            # Create GitHub release
            self.create_github_release(new_version, commits)
            
            self.logger.info(f"[PARTY] Release {new_version} completed successfully!")
            return new_version
            
        except Exception as e:
            self.logger.error(f"[X] Release failed: {str(e)}")
            if not self.dry_run:
                raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ScrambleBench Release Manager")
    parser.add_argument(
        "bump_type",
        choices=["patch", "minor", "major", "prepatch", "preminor", "premajor", "prerelease"],
        help="Type of version bump"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality checks")
    parser.add_argument("--test-pypi", action="store_true", help="Publish to Test PyPI instead of PyPI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(level=log_level, format_type="colored")
    
    # Create release manager
    release_manager = ReleaseManager(dry_run=args.dry_run)
    
    try:
        version = release_manager.full_release(
            bump_type=args.bump_type,
            skip_tests=args.skip_tests,
            skip_quality=args.skip_quality,
            test_pypi=args.test_pypi
        )
        
        print(f"\n[PARTY] Release {version} completed successfully!")
        
        if not args.dry_run:
            print("\n[CLIPBOARD] Next steps:")
            print("1. Check that the release appears on GitHub")
            print("2. Verify the package on PyPI")
            print("3. Update documentation if needed")
            print("4. Announce the release!")
        
    except KeyboardInterrupt:
        print("\n[X] Release cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Release failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()