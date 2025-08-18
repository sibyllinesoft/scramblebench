#!/usr/bin/env python3
"""
ScrambleBench CLI Demo Script

This script demonstrates the core functionality of the ScrambleBench CLI
by programmatically calling the CLI commands and showing their usage.
"""

import subprocess
import sys
import json
from pathlib import Path


def run_cli_command(command: str, capture_output: bool = True) -> str:
    """Run a CLI command and return its output."""
    try:
        result = subprocess.run(
            command.split(),
            capture_output=capture_output,
            text=True,
            check=True
        )
        return result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e.stderr}")
        return ""


def demo_language_generation():
    """Demonstrate language generation features."""
    print("🌟 Language Generation Demo")
    print("=" * 50)
    
    # Generate different types of languages
    languages = [
        ("demo_substitution", "substitution", 3),
        ("demo_phonetic", "phonetic", 5),
        ("demo_scrambled", "scrambled", 4),
        ("demo_synthetic", "synthetic", 6)
    ]
    
    for name, lang_type, complexity in languages:
        print(f"\nGenerating {lang_type} language '{name}' (complexity {complexity})...")
        command = f"scramblebench language generate {name} --type {lang_type} --complexity {complexity} --vocab-size 500 --seed 42"
        run_cli_command(command, capture_output=False)
    
    print("\n📋 Listing all generated languages:")
    output = run_cli_command("scramblebench language list --format table")
    print(output)
    
    print("\n🔍 Detailed view of synthetic language:")
    output = run_cli_command("scramblebench language show demo_synthetic --show-rules --limit 5")
    print(output)


def demo_text_transformation():
    """Demonstrate text transformation features."""
    print("\n\n🔄 Text Transformation Demo")
    print("=" * 50)
    
    sample_text = "Hello, my name is John Smith and I live in New York. I have 3 cats and 2 dogs."
    
    print(f"Original text: {sample_text}")
    
    # Transform using different strategies
    print("\n1. Proper noun replacement (random strategy):")
    output = run_cli_command(f'scramblebench transform proper-nouns "{sample_text}" --strategy random --seed 123')
    print(output)
    
    print("\n2. Synonym replacement (30% of words):")
    output = run_cli_command(f'scramblebench transform synonyms "{sample_text}" --replacement-rate 0.3 --seed 456')
    print(output)
    
    print("\n3. Language transformation (using demo_phonetic):")
    output = run_cli_command(f'scramblebench transform text "{sample_text}" demo_phonetic --preserve-numbers --preserve-proper-nouns')
    print(output)


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n\n📦 Batch Processing Demo")
    print("=" * 50)
    
    # Create sample benchmark data
    sample_data = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "category": "Geography"
        },
        {
            "question": "How many legs does a spider have?", 
            "answer": "8",
            "category": "Biology"
        },
        {
            "question": "What is 15 + 27?",
            "answer": "42", 
            "category": "Math"
        }
    ]
    
    # Ensure data directory exists
    data_dir = Path("data/benchmarks")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    sample_file = data_dir / "demo_questions.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample benchmark file: {sample_file}")
    
    print("\n📚 Extracting vocabulary from sample data:")
    output = run_cli_command(f"scramblebench batch extract-vocab {sample_file} --min-freq 1 --max-words 100")
    print(output)
    
    print("\n🔄 Transforming batch data using demo_substitution language:")
    output = run_cli_command(f"scramblebench batch transform {sample_file} demo_substitution")
    print(output)


def demo_utility_features():
    """Demonstrate utility and analysis features."""
    print("\n\n🔧 Utility Features Demo")
    print("=" * 50)
    
    print("📊 Language statistics for demo_synthetic:")
    output = run_cli_command("scramblebench util stats demo_synthetic")
    print(output)
    
    print("\n📤 Exporting rules from demo_phonetic:")
    output = run_cli_command("scramblebench util export-rules demo_phonetic --format json")
    print(output)
    
    print("\n✅ Validating transformation consistency:")
    test_text = "This is a test sentence for validation."
    output = run_cli_command(f'scramblebench util validate demo_substitution "{test_text}"')
    print(output)


def demo_json_output():
    """Demonstrate JSON output for automation."""
    print("\n\n🤖 JSON Output for Automation")
    print("=" * 50)
    
    print("Getting language list as JSON:")
    output = run_cli_command("scramblebench language list --output-format json")
    if output:
        data = json.loads(output)
        print(f"Found {len(data['languages'])} languages:")
        for lang in data['languages']:
            print(f"  - {lang['name']} ({lang['type']}) - {lang['vocab_size']} words")
    
    print("\nGetting language details as JSON:")
    output = run_cli_command("scramblebench language show demo_synthetic --output-format json")
    if output:
        data = json.loads(output)
        print(f"Language: {data['name']}")
        print(f"Type: {data['type']}")
        print(f"Vocabulary size: {data['vocab_size']}")
        print(f"Rules count: {data['rules_count']}")


def cleanup_demo():
    """Clean up demo files and languages."""
    print("\n\n🧹 Cleanup")
    print("=" * 20)
    
    # Delete demo languages
    demo_languages = ["demo_substitution", "demo_phonetic", "demo_scrambled", "demo_synthetic"]
    
    for lang in demo_languages:
        print(f"Deleting language: {lang}")
        run_cli_command(f"scramblebench language delete {lang} --force", capture_output=False)
    
    # Clean up demo files
    demo_file = Path("data/benchmarks/demo_questions.json")
    if demo_file.exists():
        demo_file.unlink()
        print(f"Deleted demo file: {demo_file}")


def main():
    """Run the complete demo."""
    print("🚀 ScrambleBench CLI Demonstration")
    print("=" * 60)
    print("This demo showcases the main features of the ScrambleBench CLI.")
    print("Make sure you have installed ScrambleBench with: pip install -e .")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return
    
    try:
        # Run all demo sections
        demo_language_generation()
        demo_text_transformation()
        demo_batch_processing()
        demo_utility_features()
        demo_json_output()
        
        print("\n\n🎉 Demo Complete!")
        print("=" * 30)
        print("You've seen the core functionality of ScrambleBench CLI:")
        print("✓ Language generation (4 types)")
        print("✓ Text transformations (proper nouns, synonyms, languages)")
        print("✓ Batch processing (vocabulary extraction, problem transformation)")
        print("✓ Utility functions (stats, export, validation)")
        print("✓ JSON output for automation")
        
        print("\nFor more details, see CLI_GUIDE.md or run:")
        print("  scramblebench --help")
        print("  scramblebench language --help")
        print("  scramblebench transform --help")
        
        # Ask about cleanup
        print("\nWould you like to clean up demo files? (y/N): ", end="")
        response = input().strip().lower()
        if response in ('y', 'yes'):
            cleanup_demo()
            print("Cleanup complete!")
        else:
            print("Demo files preserved. You can explore them or clean up manually.")
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. You may want to run cleanup:")
        print("python examples/cli_demo.py --cleanup")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Make sure ScrambleBench is properly installed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_demo()
    else:
        main()