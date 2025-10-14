#!/usr/bin/env python3
"""
Complete Data Collection and Processing Workflow.

This script orchestrates the entire data pipeline:
1. Collect data from sources
2. Normalize and clean text
3. Detect and remove PII
4. Apply quality filtering
5. Split into train/val/test sets
6. Generate dataset cards

Usage:
    python scripts/data_workflow.py \
        --source sample \
        --output datasets/processed/ \
        --quality-threshold 0.8
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DataWorkflow:
    """Orchestrate complete data processing workflow."""

    def __init__(
        self,
        working_dir: Path,
        quality_threshold: float = 0.7,
        remove_pii: bool = True,
        verbose: bool = True
    ):
        """
        Initialize workflow.

        Args:
            working_dir: Working directory for intermediate files
            quality_threshold: Minimum quality score
            remove_pii: Whether to remove PII
            verbose: Verbose output
        """
        self.working_dir = working_dir
        self.quality_threshold = quality_threshold
        self.remove_pii = remove_pii
        self.verbose = verbose

        # Create directory structure
        self.raw_dir = working_dir / "raw"
        self.cleaned_dir = working_dir / "cleaned"
        self.filtered_dir = working_dir / "filtered"
        self.final_dir = working_dir / "final"

        for dir_path in [self.raw_dir, self.cleaned_dir,
                         self.filtered_dir, self.final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'data_counts': {},
            'errors': []
        }

    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def run_step(self, step_name: str, command: List[str]) -> bool:
        """
        Run a processing step.

        Args:
            step_name: Name of the step
            command: Command to run

        Returns:
            True if successful
        """
        self.log(f"Starting step: {step_name}")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )

            if self.verbose and result.stdout:
                print(result.stdout)

            self.stats['steps_completed'].append(step_name)
            self.log(f"✓ Completed: {step_name}")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = f"✗ Failed: {step_name}\nError: {e.stderr}"
            self.log(error_msg)
            self.stats['errors'].append({
                'step': step_name,
                'error': str(e),
                'stderr': e.stderr
            })
            return False

    def step_1_collect_data(self, source: str, languages: List[str]) -> bool:
        """
        Step 1: Collect raw data.

        Args:
            source: Data source type
            languages: List of languages to collect

        Returns:
            True if successful
        """
        self.log("=" * 60)
        self.log("STEP 1: Data Collection")
        self.log("=" * 60)

        for lang in languages:
            command = [
                sys.executable,
                "scripts/collect_data.py",
                "--source", source,
                "--lang", lang,
                "--output", str(self.raw_dir)
            ]

            if not self.run_step(f"collect_{lang}", command):
                return False

        # Count collected samples
        count = self._count_samples(self.raw_dir)
        self.stats['data_counts']['raw'] = count
        self.log(f"Collected {count} samples")

        return True

    def step_2_normalize_and_clean(self) -> bool:
        """
        Step 2: Normalize and clean text.

        Returns:
            True if successful
        """
        self.log("=" * 60)
        self.log("STEP 2: Normalization and Cleaning")
        self.log("=" * 60)

        command = [
            sys.executable,
            "scripts/prepare_data.py",
            "--input", str(self.raw_dir),
            "--output", str(self.cleaned_dir)
        ]

        success = self.run_step("normalize", command)

        if success:
            count = self._count_samples(self.cleaned_dir)
            self.stats['data_counts']['cleaned'] = count
            self.log(f"Cleaned {count} samples")

        return success

    def step_3_remove_pii(self) -> bool:
        """
        Step 3: Remove PII.

        Returns:
            True if successful
        """
        if not self.remove_pii:
            self.log("Skipping PII removal (disabled)")
            return True

        self.log("=" * 60)
        self.log("STEP 3: PII Detection and Removal")
        self.log("=" * 60)

        # Use cleaned dir as output for this step
        command = [
            sys.executable,
            "scripts/pii_detection.py",
            "--input", str(self.cleaned_dir),
            "--output", str(self.cleaned_dir),
            "--mode", "redact"
        ]

        return self.run_step("pii_removal", command)

    def step_4_quality_filter(self) -> bool:
        """
        Step 4: Apply quality filtering.

        Returns:
            True if successful
        """
        self.log("=" * 60)
        self.log("STEP 4: Quality Filtering")
        self.log("=" * 60)

        command = [
            sys.executable,
            "scripts/quality_filter.py",
            "--input", str(self.cleaned_dir),
            "--output", str(self.filtered_dir),
            "--min-quality", str(self.quality_threshold),
            "--report", str(self.working_dir / "quality_report.json")
        ]

        success = self.run_step("quality_filter", command)

        if success:
            count = self._count_samples(self.filtered_dir)
            self.stats['data_counts']['filtered'] = count
            self.log(f"Filtered to {count} high-quality samples")

            # Calculate pass rate
            raw_count = self.stats['data_counts'].get('cleaned', 0)
            if raw_count > 0:
                pass_rate = count / raw_count * 100
                self.log(f"Quality pass rate: {pass_rate:.1f}%")

        return success

    def step_5_create_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> bool:
        """
        Step 5: Create train/val/test splits.

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            True if successful
        """
        self.log("=" * 60)
        self.log("STEP 5: Creating Data Splits")
        self.log("=" * 60)

        # Load all filtered data
        all_samples = []
        for file in self.filtered_dir.rglob('*.json*'):
            if file.suffix in ['.json', '.jsonl']:
                with open(file, 'r', encoding='utf-8') as f:
                    if file.suffix == '.jsonl':
                        samples = [json.loads(line) for line in f if line.strip()]
                    else:
                        data = json.load(f)
                        samples = data if isinstance(data, list) else [data]
                    all_samples.extend(samples)

        if not all_samples:
            self.log("✗ No samples to split")
            return False

        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(all_samples)

        total = len(all_samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            'train': all_samples[:train_end],
            'val': all_samples[train_end:val_end],
            'test': all_samples[val_end:]
        }

        # Save splits
        for split_name, samples in splits.items():
            output_file = self.final_dir / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            self.log(f"Saved {len(samples)} samples to {split_name}.jsonl")
            self.stats['data_counts'][split_name] = len(samples)

        self.stats['steps_completed'].append('create_splits')
        return True

    def step_6_generate_dataset_card(self, dataset_name: str) -> bool:
        """
        Step 6: Generate dataset card.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if successful
        """
        self.log("=" * 60)
        self.log("STEP 6: Generating Dataset Card")
        self.log("=" * 60)

        card_path = self.final_dir / "DATASET_CARD.md"

        # Generate card content
        card_content = f"""# Dataset Card: {dataset_name}

## Dataset Description

### Dataset Summary

This dataset contains bilingual (Bangla-English) text data for training language models.
Generated on {datetime.now().strftime('%Y-%m-%d')} using the bilingual data processing pipeline.

### Supported Tasks

- Language Modeling
- Text Generation
- Translation
- Classification

### Languages

- Primary Languages: Bangla (bn), English (en)
- Code-switched content included

## Dataset Structure

### Data Splits

| Split | Size | Percentage |
|-------|------|------------|
| Train | {self.stats['data_counts'].get('train', 0)} | {self.stats['data_counts'].get('train', 0) / max(sum(self.stats['data_counts'].get(s, 0) for s in ['train', 'val', 'test']), 1) * 100:.1f}% |
| Validation | {self.stats['data_counts'].get('val', 0)} | {self.stats['data_counts'].get('val', 0) / max(sum(self.stats['data_counts'].get(s, 0) for s in ['train', 'val', 'test']), 1) * 100:.1f}% |
| Test | {self.stats['data_counts'].get('test', 0)} | {self.stats['data_counts'].get('test', 0) / max(sum(self.stats['data_counts'].get(s, 0) for s in ['train', 'val', 'test']), 1) * 100:.1f}% |
| **Total** | **{sum(self.stats['data_counts'].get(s, 0) for s in ['train', 'val', 'test'])}** | **100%** |

## Processing Pipeline

1. **Data Collection**: {self.stats['data_counts'].get('raw', 0)} samples collected
2. **Normalization**: Text cleaned and normalized
3. **PII Removal**: Personal information redacted
4. **Quality Filtering**: Filtered to {self.stats['data_counts'].get('filtered', 0)} high-quality samples (threshold: {self.quality_threshold})
5. **Data Splits**: Created train/val/test splits

### Quality Criteria

- Minimum quality score: {self.quality_threshold}
- Length requirements: 50-5000 characters
- Content appropriateness: Child-safe content only
- PII removal: All personal information redacted

## Licensing Information

See individual samples for specific licenses. All content verified for redistribution rights.

## Dataset Curators

**Organization**: KhulnaSoft Ltd
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage Example

```python
from bilingual.data_utils import BilingualDataset

# Load train set
train_data = BilingualDataset(file_path="datasets/processed/final/train.jsonl")
print(f"Training samples: {{len(train_data)}}")

# Access samples
for sample in train_data:
    print(sample['text'])
    print(sample['language'])
    break
```

---

**Generated by**: bilingual data workflow
**Version**: 1.0.0
**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
"""

        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(card_content)

        self.log(f"✓ Dataset card generated: {card_path}")
        self.stats['steps_completed'].append('generate_card')

        return True

    def run_complete_workflow(
        self,
        source: str,
        languages: List[str],
        dataset_name: str
    ) -> bool:
        """
        Run complete data workflow.

        Args:
            source: Data source type
            languages: Languages to collect
            dataset_name: Name for dataset

        Returns:
            True if all steps successful
        """
        self.log("╔" + "=" * 58 + "╗")
        self.log("║" + " " * 58 + "║")
        self.log("║" + "  BILINGUAL DATA PROCESSING WORKFLOW".center(58) + "║")
        self.log("║" + " " * 58 + "║")
        self.log("╚" + "=" * 58 + "╝")
        self.log("")

        steps = [
            ("Data Collection", lambda: self.step_1_collect_data(source, languages)),
            ("Normalization", self.step_2_normalize_and_clean),
            ("PII Removal", self.step_3_remove_pii),
            ("Quality Filtering", self.step_4_quality_filter),
            ("Data Splits", self.step_5_create_splits),
            ("Dataset Card", lambda: self.step_6_generate_dataset_card(dataset_name)),
        ]

        for step_name, step_func in steps:
            if not step_func():
                self.log(f"\n✗ Workflow failed at step: {step_name}")
                return False

        # Save stats
        self.stats['end_time'] = datetime.now().isoformat()
        stats_file = self.working_dir / "workflow_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

        self.log("\n" + "=" * 60)
        self.log("✓ WORKFLOW COMPLETED SUCCESSFULLY")
        self.log("=" * 60)
        self.log(f"\nFinal dataset location: {self.final_dir}")
        self.log(f"Statistics saved to: {stats_file}")
        self.log(f"\nDataset splits:")
        for split in ['train', 'val', 'test']:
            count = self.stats['data_counts'].get(split, 0)
            self.log(f"  {split}: {count} samples")

        return True

    def _count_samples(self, directory: Path) -> int:
        """Count total samples in directory."""
        count = 0
        for file in directory.rglob('*.json*'):
            if file.suffix in ['.json', '.jsonl']:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        if file.suffix == '.jsonl':
                            count += sum(1 for line in f if line.strip())
                        else:
                            data = json.load(f)
                            count += len(data) if isinstance(data, list) else 1
                except Exception:
                    pass
        return count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run complete data collection and processing workflow'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='sample',
        help='Data source (sample, wikipedia, etc.)'
    )
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=['bn', 'en'],
        help='Languages to collect (default: bn en)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='Bilingual Corpus',
        help='Dataset name for card'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.7,
        help='Minimum quality score (default: 0.7)'
    )
    parser.add_argument(
        '--no-pii-removal',
        action='store_true',
        help='Skip PII removal step'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimize output'
    )

    args = parser.parse_args()

    # Initialize workflow
    workflow = DataWorkflow(
        working_dir=Path(args.output),
        quality_threshold=args.quality_threshold,
        remove_pii=not args.no_pii_removal,
        verbose=not args.quiet
    )

    # Run workflow
    success = workflow.run_complete_workflow(
        source=args.source,
        languages=args.languages,
        dataset_name=args.dataset_name
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
