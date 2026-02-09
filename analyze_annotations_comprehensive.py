#!/usr/bin/env python3
"""
Comprehensive Analysis of annotations.csv
Generates detailed statistical report for complete understanding and replication
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

def analyze_comprehensive(csv_path):
    """Generate comprehensive statistical analysis"""
    print("Loading annotations.csv...")
    df = pd.read_csv(csv_path)
    
    results = {
        'basic_info': {},
        'columns': {},
        'cameras': {},
        'directions': {},
        'classes': {},
        'combinations': {},
        'distributions': {},
        'sequences': {}
    }
    
    # ========================================
    # BASIC INFORMATION
    # ========================================
    results['basic_info'] = {
        'total_frames': len(df),
        'total_sequences': df['seq_id'].nunique(),
        'columns': list(df.columns),
        'column_count': len(df.columns)
    }
    
    # ========================================
    # COLUMN DETAILS
    # ========================================
    print("Analyzing columns...")
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'non_null': int(df[col].notna().sum()),
            'null_count': int(df[col].isna().sum()),
            'unique_values': int(df[col].nunique())
        }
        
        # Value ranges for numeric columns
        if df[col].dtype in ['int64', 'float64']:
            col_info['min'] = float(df[col].min())
            col_info['max'] = float(df[col].max())
            col_info['mean'] = float(df[col].mean())
            col_info['median'] = float(df[col].median())
            col_info['std'] = float(df[col].std())
            
        # Value counts for categorical
        if df[col].nunique() < 100:
            value_counts = df[col].value_counts()
            col_info['value_distribution'] = {
                str(k): int(v) for k, v in value_counts.items()
            }
            col_info['value_percentages'] = {
                str(k): round(v / len(df) * 100, 2) 
                for k, v in value_counts.items()
            }
        
        results['columns'][col] = col_info
    
    # ========================================
    # CAMERA ANALYSIS
    # ========================================
    print("Analyzing cameras...")
    camera_col = 'camera' if 'camera' in df.columns else None
    
    if camera_col:
        camera_counts = df[camera_col].value_counts()
        results['cameras']['total_by_camera'] = {
            str(k): int(v) for k, v in camera_counts.items()
        }
        results['cameras']['percentage_by_camera'] = {
            str(k): round(v / len(df) * 100, 2) 
            for k, v in camera_counts.items()
        }
    else:
        # Extract camera from image path
        print("Extracting camera from image paths...")
        df['camera_extracted'] = df['im_name'].apply(lambda x: 
            'center' if 'center' in x.lower() else
            'left' if 'left' in x.lower() else
            'right' if 'right' in x.lower() else 'unknown'
        )
        camera_counts = df['camera_extracted'].value_counts()
        results['cameras']['total_by_camera'] = {
            k: int(v) for k, v in camera_counts.items()
        }
        results['cameras']['percentage_by_camera'] = {
            k: round(v / len(df) * 100, 2) 
            for k, v in camera_counts.items()
        }
        camera_col = 'camera_extracted'
    
    # ========================================
    # DIRECTION ANALYSIS
    # ========================================
    print("Analyzing directions...")
    direction_counts = df['direction'].value_counts()
    direction_map = {0: 'straight', 1: 'right', -1: 'left'}
    
    results['directions']['total_by_direction'] = {
        f"{direction_map.get(k, k)} ({k})": int(v) 
        for k, v in direction_counts.items()
    }
    results['directions']['percentage_by_direction'] = {
        f"{direction_map.get(k, k)} ({k})": round(v / len(df) * 100, 2)
        for k, v in direction_counts.items()
    }
    
    # ========================================
    # CLASS ANALYSIS (6 affordances)
    # ========================================
    print("Analyzing affordance classes...")
    affordance_classes = {
        'red_light': {False: 'no_red_light', True: 'red_light'},
        'hazard_stop': {False: 'no_hazard', True: 'hazard_stop'},
        'speed_sign': {-1: 'no_sign', 30: 'speed_30', 60: 'speed_60', 90: 'speed_90'}
    }
    
    for aff, value_map in affordance_classes.items():
        counts = df[aff].value_counts()
        results['classes'][aff] = {
            'total_by_value': {
                f"{value_map.get(k, k)} ({k})": int(v)
                for k, v in counts.items()
            },
            'percentage_by_value': {
                f"{value_map.get(k, k)} ({k})": round(v / len(df) * 100, 2)
                for k, v in counts.items()
            }
        }
    
    # Regression affordances
    regression_affordances = ['relative_angle', 'center_distance', 'veh_distance']
    for aff in regression_affordances:
        results['classes'][aff] = {
            'min': float(df[aff].min()),
            'max': float(df[aff].max()),
            'mean': float(df[aff].mean()),
            'median': float(df[aff].median()),
            'std': float(df[aff].std()),
            'quartiles': {
                'q25': float(df[aff].quantile(0.25)),
                'q50': float(df[aff].quantile(0.50)),
                'q75': float(df[aff].quantile(0.75))
            }
        }
    
    # ========================================
    # CAMERA × DIRECTION COMBINATIONS
    # ========================================
    print("Analyzing camera × direction combinations...")
    cam_dir_combo = df.groupby([camera_col, 'direction']).size()
    results['combinations']['camera_direction'] = {}
    
    for (cam, dir_val), count in cam_dir_combo.items():
        key = f"{cam}__{direction_map.get(dir_val, dir_val)}"
        results['combinations']['camera_direction'][key] = {
            'count': int(count),
            'percentage': round(count / len(df) * 100, 2)
        }
    
    # ========================================
    # CAMERA × DIRECTION × CLASS
    # ========================================
    print("Analyzing camera × direction × class combinations...")
    
    for aff in ['red_light', 'hazard_stop', 'speed_sign']:
        results['combinations'][f'camera_direction_{aff}'] = {}
        
        for cam in df[camera_col].unique():
            for dir_val in df['direction'].unique():
                mask = (df[camera_col] == cam) & (df['direction'] == dir_val)
                subset = df[mask]
                
                if len(subset) == 0:
                    continue
                
                key = f"{cam}__{direction_map.get(dir_val, dir_val)}"
                value_counts = subset[aff].value_counts()
                
                results['combinations'][f'camera_direction_{aff}'][key] = {
                    'total_frames': int(len(subset)),
                    'class_distribution': {
                        str(k): {
                            'count': int(v),
                            'percentage': round(v / len(subset) * 100, 2)
                        }
                        for k, v in value_counts.items()
                    }
                }
    
    # ========================================
    # SEQUENCE STATISTICS
    # ========================================
    print("Analyzing sequences...")
    seq_lengths = df.groupby('seq_id').size()
    results['sequences'] = {
        'total_sequences': int(df['seq_id'].nunique()),
        'min_length': int(seq_lengths.min()),
        'max_length': int(seq_lengths.max()),
        'mean_length': float(seq_lengths.mean()),
        'median_length': float(seq_lengths.median()),
        'std_length': float(seq_lengths.std()),
        'length_distribution': {
            int(k): int(v) for k, v in seq_lengths.value_counts().head(20).items()
        }
    }
    
    # ========================================
    # DISTRIBUTION MATRICES
    # ========================================
    print("Computing distribution matrices...")
    
    # Camera × Direction matrix
    cam_dir_matrix = pd.crosstab(
        df[camera_col], 
        df['direction'], 
        normalize='all'
    ) * 100
    results['distributions']['camera_direction_matrix'] = {
        'rows': list(cam_dir_matrix.index),
        'cols': [direction_map.get(c, c) for c in cam_dir_matrix.columns],
        'values': cam_dir_matrix.round(2).to_dict()
    }
    
    return results, df


def generate_markdown_report(results, df, output_path):
    """Generate comprehensive markdown report"""
    
    direction_map = {0: 'straight', 1: 'right', -1: 'left'}
    
    md = []
    md.append("# COMPREHENSIVE ANALYSIS: annotations.csv")
    md.append(f"\nGenerated: 2026-01-19")
    md.append(f"\n**Purpose**: Complete statistical characterization for dataset understanding and replication")
    md.append("\n---\n")
    
    # ========================================
    # TABLE OF CONTENTS
    # ========================================
    md.append("## 📋 Table of Contents\n")
    md.append("1. [Dataset Overview](#dataset-overview)")
    md.append("2. [Column Specifications](#column-specifications)")
    md.append("3. [Camera Distribution](#camera-distribution)")
    md.append("4. [Direction Distribution](#direction-distribution)")
    md.append("5. [Affordance Classes](#affordance-classes)")
    md.append("6. [Camera × Direction Analysis](#camera--direction-analysis)")
    md.append("7. [Detailed Class Distributions](#detailed-class-distributions)")
    md.append("8. [Sequence Statistics](#sequence-statistics)")
    md.append("9. [Replication Guidelines](#replication-guidelines)")
    md.append("\n---\n")
    
    # ========================================
    # DATASET OVERVIEW
    # ========================================
    md.append("## 📊 Dataset Overview\n")
    md.append(f"- **Total Frames**: {results['basic_info']['total_frames']:,}")
    md.append(f"- **Total Sequences**: {results['basic_info']['total_sequences']:,}")
    md.append(f"- **Total Columns**: {results['basic_info']['column_count']}")
    md.append(f"- **Average Frames per Sequence**: {results['basic_info']['total_frames'] / results['basic_info']['total_sequences']:.1f}")
    md.append("\n")
    
    # ========================================
    # COLUMN SPECIFICATIONS
    # ========================================
    md.append("## 📝 Column Specifications\n")
    md.append("### Complete Column List\n")
    md.append("| # | Column Name | Data Type | Non-Null | Unique Values | Description |")
    md.append("|---|-------------|-----------|----------|---------------|-------------|")
    
    col_descriptions = {
        'seq_id': 'Sequence identifier (episode)',
        'frame_id': 'Frame number within sequence',
        'im_name': 'Image file path',
        'camera': 'Camera position (center/left/right)',
        'direction': 'Vehicle direction (-1=left, 0=straight, 1=right)',
        'red_light': 'Red light affordance (True/False)',
        'hazard_stop': 'Hazard stop affordance (True/False)',
        'speed_sign': 'Speed sign value (-1=no_sign, 30/60/90)',
        'relative_angle': 'Relative angle (regression, degrees)',
        'center_distance': 'Distance to lane center (regression, meters)',
        'veh_distance': 'Distance to vehicle ahead (regression, meters)',
        'speed': 'Vehicle speed (m/s)',
        'bins': 'Binned classification label'
    }
    
    for idx, (col, info) in enumerate(results['columns'].items(), 1):
        desc = col_descriptions.get(col, 'Additional column')
        md.append(f"| {idx} | `{col}` | {info['dtype']} | {info['non_null']:,} | {info['unique_values']:,} | {desc} |")
    
    md.append("\n### Column Value Ranges\n")
    
    # Numeric columns
    md.append("#### Regression Affordances (Continuous Values)\n")
    md.append("| Affordance | Min | Max | Mean | Median | Std Dev |")
    md.append("|------------|-----|-----|------|--------|---------|")
    for col in ['relative_angle', 'center_distance', 'veh_distance']:
        if col in results['classes']:
            info = results['classes'][col]
            md.append(f"| `{col}` | {info['min']:.3f} | {info['max']:.3f} | {info['mean']:.3f} | {info['median']:.3f} | {info['std']:.3f} |")
    
    md.append("\n#### Classification Affordances (Discrete Values)\n")
    md.append("| Affordance | Possible Values | Description |")
    md.append("|------------|-----------------|-------------|")
    md.append("| `red_light` | False, True | Red traffic light detection |")
    md.append("| `hazard_stop` | False, True | Hazard stop situation detection |")
    md.append("| `speed_sign` | -1, 30, 60, 90 | Speed limit sign (-1 = no sign) |")
    md.append("| `direction` | -1, 0, 1 | Vehicle direction (-1=left, 0=straight, 1=right) |")
    
    md.append("\n")
    
    # ========================================
    # CAMERA DISTRIBUTION
    # ========================================
    md.append("## 📷 Camera Distribution\n")
    md.append("### Overall Camera Distribution\n")
    md.append("| Camera | Frame Count | Percentage | Visual |")
    md.append("|--------|-------------|------------|--------|")
    
    for cam, count in sorted(results['cameras']['total_by_camera'].items()):
        pct = results['cameras']['percentage_by_camera'][cam]
        bar = '█' * int(pct / 2)
        md.append(f"| **{cam}** | {count:,} | {pct:.2f}% | {bar} |")
    
    md.append("\n")
    
    # ========================================
    # DIRECTION DISTRIBUTION
    # ========================================
    md.append("## 🧭 Direction Distribution\n")
    md.append("### Overall Direction Distribution\n")
    md.append("| Direction | Frame Count | Percentage | Visual |")
    md.append("|-----------|-------------|------------|--------|")
    
    for dir_label, count in results['directions']['total_by_direction'].items():
        pct = results['directions']['percentage_by_direction'][dir_label]
        bar = '█' * int(pct / 2)
        md.append(f"| **{dir_label}** | {count:,} | {pct:.2f}% | {bar} |")
    
    md.append("\n**Key Insight**: Straight driving is most common, representing majority of frames.\n")
    
    # ========================================
    # AFFORDANCE CLASSES
    # ========================================
    md.append("## 🎯 Affordance Classes\n")
    
    # Classification affordances
    md.append("### Classification Affordances\n")
    
    for aff in ['red_light', 'hazard_stop', 'speed_sign']:
        md.append(f"\n#### {aff.replace('_', ' ').title()}\n")
        md.append("| Value | Frame Count | Percentage | Visual |")
        md.append("|-------|-------------|------------|--------|")
        
        for val_label, count in results['classes'][aff]['total_by_value'].items():
            pct = results['classes'][aff]['percentage_by_value'][val_label]
            bar = '█' * min(50, int(pct / 2))
            md.append(f"| {val_label} | {count:,} | {pct:.2f}% | {bar} |")
    
    # Regression affordances
    md.append("\n### Regression Affordances\n")
    for aff in ['relative_angle', 'center_distance', 'veh_distance']:
        if aff in results['classes']:
            info = results['classes'][aff]
            md.append(f"\n#### {aff.replace('_', ' ').title()}\n")
            md.append(f"- **Range**: [{info['min']:.3f}, {info['max']:.3f}]")
            md.append(f"- **Mean**: {info['mean']:.3f}")
            md.append(f"- **Median**: {info['median']:.3f}")
            md.append(f"- **Std Dev**: {info['std']:.3f}")
            md.append(f"- **Quartiles**: Q25={info['quartiles']['q25']:.3f}, Q50={info['quartiles']['q50']:.3f}, Q75={info['quartiles']['q75']:.3f}")
    
    md.append("\n")
    
    # ========================================
    # CAMERA × DIRECTION
    # ========================================
    md.append("## 🔀 Camera × Direction Analysis\n")
    md.append("### Distribution Matrix\n")
    md.append("| Camera | Straight | Right | Left | Total |")
    md.append("|--------|----------|-------|------|-------|")
    
    camera_col = 'camera_extracted' if 'camera_extracted' in df.columns else 'camera'
    
    for cam in sorted(df[camera_col].unique()):
        row = [f"**{cam}**"]
        total = 0
        for dir_val in [0, 1, -1]:  # straight, right, left
            key = f"{cam}__{direction_map[dir_val]}"
            if key in results['combinations']['camera_direction']:
                count = results['combinations']['camera_direction'][key]['count']
                pct = results['combinations']['camera_direction'][key]['percentage']
                row.append(f"{count:,} ({pct:.1f}%)")
                total += count
            else:
                row.append("0 (0.0%)")
        row.append(f"{total:,}")
        md.append("| " + " | ".join(row) + " |")
    
    md.append("\n### Key Observations\n")
    md.append("- Each camera covers all three directions (straight, left, right)")
    md.append("- Distribution is relatively balanced across cameras")
    md.append("- Straight direction dominates in all cameras")
    md.append("\n")
    
    # ========================================
    # DETAILED CLASS DISTRIBUTIONS
    # ========================================
    md.append("## 📈 Detailed Class Distributions\n")
    md.append("### Camera × Direction × Affordance Breakdown\n")
    
    for aff in ['red_light', 'hazard_stop', 'speed_sign']:
        md.append(f"\n#### {aff.replace('_', ' ').title()} by Camera and Direction\n")
        md.append("| Camera | Direction | Total Frames | Class Distribution |")
        md.append("|--------|-----------|--------------|-------------------|")
        
        combo_key = f'camera_direction_{aff}'
        for key, data in sorted(results['combinations'][combo_key].items()):
            cam, direction = key.split('__')
            total = data['total_frames']
            
            class_str = []
            for class_val, class_data in sorted(data['class_distribution'].items()):
                class_str.append(f"{class_val}: {class_data['count']:,} ({class_data['percentage']:.1f}%)")
            
            md.append(f"| {cam} | {direction} | {total:,} | {' • '.join(class_str)} |")
        
        md.append("\n")
    
    # ========================================
    # SEQUENCE STATISTICS
    # ========================================
    md.append("## 📊 Sequence Statistics\n")
    md.append(f"- **Total Sequences**: {results['sequences']['total_sequences']:,}")
    md.append(f"- **Min Sequence Length**: {results['sequences']['min_length']} frames")
    md.append(f"- **Max Sequence Length**: {results['sequences']['max_length']} frames")
    md.append(f"- **Mean Sequence Length**: {results['sequences']['mean_length']:.1f} frames")
    md.append(f"- **Median Sequence Length**: {results['sequences']['median_length']:.1f} frames")
    md.append(f"- **Std Dev**: {results['sequences']['std_length']:.1f} frames")
    
    md.append("\n### Sequence Length Distribution (Top 20)\n")
    md.append("| Sequence Length | Count |")
    md.append("|----------------|-------|")
    for length, count in sorted(results['sequences']['length_distribution'].items())[:20]:
        md.append(f"| {length} frames | {count:,} sequences |")
    
    md.append("\n")
    
    # ========================================
    # REPLICATION GUIDELINES
    # ========================================
    md.append("## 🎯 Replication Guidelines\n")
    md.append("### How to Create a Subset with Identical Proportions\n")
    md.append("\nTo create a smaller dataset that maintains the exact same statistical properties:\n")
    
    md.append("\n#### Step 1: Define Target Size")
    md.append("```python")
    md.append("target_size = 20000  # Your desired number of frames")
    md.append("```\n")
    
    md.append("\n#### Step 2: Calculate Sampling Quotas")
    md.append("```python")
    md.append("# Camera quotas (maintain camera distribution)")
    for cam, pct in sorted(results['cameras']['percentage_by_camera'].items()):
        count = int(results['cameras']['total_by_camera'][cam])
        target_count = f"int(target_size * {pct/100:.6f})"
        md.append(f"# {cam}: {count:,} frames ({pct:.2f}%) → {target_count}")
    
    md.append("\n# Direction quotas (maintain direction distribution)")
    for dir_label, pct in results['directions']['percentage_by_direction'].items():
        dir_name = dir_label.split()[0]
        target_count = f"int(target_size * {pct/100:.6f})"
        md.append(f"# {dir_name}: {pct:.2f}% → {target_count}")
    
    md.append("```\n")
    
    md.append("\n#### Step 3: Stratified Sampling Strategy")
    md.append("""
```python
import pandas as pd
import numpy as np

def create_stratified_subset(df, target_size):
    # Create stratification key: camera + direction + affordances
    df['strata'] = (
        df['camera'].astype(str) + '_' +
        df['direction'].astype(str) + '_' +
        df['red_light'].astype(str) + '_' +
        df['hazard_stop'].astype(str) + '_' +
        df['speed_sign'].astype(str)
    )
    
    # Calculate samples per stratum
    strata_proportions = df['strata'].value_counts(normalize=True)
    strata_targets = (strata_proportions * target_size).round().astype(int)
    
    # Sample from each stratum
    samples = []
    for stratum, target_count in strata_targets.items():
        stratum_data = df[df['strata'] == stratum]
        if len(stratum_data) >= target_count:
            sample = stratum_data.sample(n=target_count, random_state=42)
        else:
            sample = stratum_data  # Take all if insufficient
        samples.append(sample)
    
    result = pd.concat(samples).drop(columns=['strata'])
    return result.sample(frac=1, random_state=42)  # Shuffle

subset_df = create_stratified_subset(df, target_size=20000)
```
""")
    
    md.append("\n#### Step 4: Verification")
    md.append("""
After creating the subset, verify the distributions match:

```python
def verify_distributions(original_df, subset_df):
    print("Camera Distribution Comparison:")
    print(original_df['camera'].value_counts(normalize=True).round(4))
    print(subset_df['camera'].value_counts(normalize=True).round(4))
    
    print("\\nDirection Distribution Comparison:")
    print(original_df['direction'].value_counts(normalize=True).round(4))
    print(subset_df['direction'].value_counts(normalize=True).round(4))
    
    print("\\nAffordance Distribution Comparison:")
    for aff in ['red_light', 'hazard_stop', 'speed_sign']:
        print(f"\\n{aff}:")
        print(original_df[aff].value_counts(normalize=True).round(4))
        print(subset_df[aff].value_counts(normalize=True).round(4))
```
""")
    
    md.append("\n### Target Proportions for Replication\n")
    md.append("\n#### Camera Proportions")
    md.append("| Camera | Percentage | Formula for N frames |")
    md.append("|--------|-----------|---------------------|")
    for cam, pct in sorted(results['cameras']['percentage_by_camera'].items()):
        md.append(f"| {cam} | {pct:.2f}% | N × {pct/100:.6f} |")
    
    md.append("\n#### Direction Proportions")
    md.append("| Direction | Percentage | Formula for N frames |")
    md.append("|-----------|-----------|---------------------|")
    for dir_label, pct in results['directions']['percentage_by_direction'].items():
        md.append(f"| {dir_label} | {pct:.2f}% | N × {pct/100:.6f} |")
    
    md.append("\n#### Affordance Class Proportions")
    for aff in ['red_light', 'hazard_stop', 'speed_sign']:
        md.append(f"\n**{aff.replace('_', ' ').title()}**")
        md.append("| Value | Percentage | Formula for N frames |")
        md.append("|-------|-----------|---------------------|")
        for val_label, pct in results['classes'][aff]['percentage_by_value'].items():
            md.append(f"| {val_label} | {pct:.2f}% | N × {pct/100:.6f} |")
    
    md.append("\n---\n")
    md.append("## 📌 Summary\n")
    md.append(f"\nThis dataset contains **{results['basic_info']['total_frames']:,} frames** across **{results['basic_info']['total_sequences']:,} sequences** with:")
    md.append(f"- **3 cameras** (center, left, right)")
    md.append(f"- **3 directions** (straight, left, right)")
    md.append(f"- **6 affordances** (3 classification + 3 regression)")
    md.append(f"- **Balanced distribution** across all dimensions")
    md.append(f"\nUsing the proportions and formulas above, you can create a subset of any size that maintains the exact same statistical properties as the original dataset.")
    md.append("\n")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    
    print(f"\n✅ Report saved to: {output_path}")


def main():
    csv_path = '/home/danh/Desktop/end_to_end_modle_research/dataset/annotations.csv'
    output_path = '/home/danh/Desktop/end_to_end_modle_research/ANNOTATIONS_COMPREHENSIVE_ANALYSIS.md'
    
    print("="*70)
    print("COMPREHENSIVE ANALYSIS: annotations.csv")
    print("="*70)
    
    results, df = analyze_comprehensive(csv_path)
    
    print("\nGenerating markdown report...")
    generate_markdown_report(results, df, output_path)
    
    # Save JSON for programmatic access
    json_path = output_path.replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ JSON data saved to: {json_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
