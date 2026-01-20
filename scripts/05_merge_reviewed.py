#!/usr/bin/env python3
"""
Script 05: Merge reviewed labels từ Label Studio

Import annotations đã review và merge với auto labels.

Usage:
    python scripts/05_merge_reviewed.py --auto data/processed/labeled --reviewed data/labelstudio/export.json --output data/processed/final
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_labelstudio_export(export_path: str) -> Dict[Tuple[str, int], Dict]:
    """
    Load Label Studio export và tạo mapping.
    
    Returns:
        Dict[(source_file, segment_id)] -> annotation
    """
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    annotations = {}
    
    for task in data:
        task_data = task.get("data", {})
        source_file = task_data.get("source_file", "")
        segment_id = task_data.get("segment_id", 0)
        
        # Get annotations
        annotation = {}
        
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("from_name") == "turn_label":
                    choices = result.get("value", {}).get("choices", [])
                    if choices:
                        annotation["reviewed_label"] = choices[0]
                
                elif result.get("from_name") == "issues":
                    choices = result.get("value", {}).get("choices", [])
                    if choices:
                        annotation["issues"] = choices
                
                elif result.get("from_name") == "notes":
                    text = result.get("value", {}).get("text", [""])[0]
                    if text:
                        annotation["notes"] = text
        
        if annotation:
            annotations[(source_file, segment_id)] = annotation
    
    return annotations


def merge_labels(
    auto_dir: str,
    reviewed_export: str,
    output_dir: str,
    prefer_reviewed: bool = True
) -> Dict:
    """
    Merge auto labels với reviewed labels.
    
    Args:
        auto_dir: Thư mục chứa auto-labeled JSON
        reviewed_export: Label Studio export JSON
        output_dir: Output directory
        prefer_reviewed: Nếu có reviewed label, dùng reviewed
    
    Returns:
        Stats dict
    """
    auto_path = Path(auto_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load reviewed
    print(f"📥 Loading reviewed annotations from {reviewed_export}")
    reviewed = load_labelstudio_export(reviewed_export)
    print(f"   Found {len(reviewed)} reviewed segments")
    
    stats = {
        "total_segments": 0,
        "reviewed": 0,
        "auto_only": 0,
        "label_changes": 0,
        "with_issues": 0,
        "final_labels": defaultdict(int)
    }
    
    # Process each file
    for json_file in sorted(auto_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        source_file = json_file.name
        
        for segment in data.get("segments", []):
            stats["total_segments"] += 1
            segment_id = segment.get("id", 0)
            
            key = (source_file, segment_id)
            
            if key in reviewed:
                ann = reviewed[key]
                stats["reviewed"] += 1
                
                # Update with reviewed label
                old_label = segment.get("auto_label", "")
                new_label = ann.get("reviewed_label", old_label)
                
                if old_label != new_label:
                    stats["label_changes"] += 1
                
                segment["label"] = new_label
                segment["reviewed"] = True
                segment["auto_label"] = old_label
                
                # Add issues/notes
                if "issues" in ann:
                    segment["issues"] = ann["issues"]
                    stats["with_issues"] += 1
                if "notes" in ann:
                    segment["notes"] = ann["notes"]
            
            else:
                # Use auto label
                segment["label"] = segment.get("auto_label", "YIELD")
                segment["reviewed"] = False
                stats["auto_only"] += 1
            
            # Count final labels
            stats["final_labels"][segment["label"]] += 1
        
        # Update stats
        data["merge_stats"] = {
            "reviewed_count": sum(1 for s in data["segments"] if s.get("reviewed")),
            "total_count": len(data["segments"])
        }
        
        # Save
        output_file = output_path / source_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Convert defaultdict to dict for JSON
    stats["final_labels"] = dict(stats["final_labels"])
    
    return stats


def validate_merged(output_dir: str) -> List[str]:
    """Validate merged data và trả về warnings"""
    warnings = []
    
    for json_file in Path(output_dir).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for segment in data.get("segments", []):
            text = segment.get("text", "")
            label = segment.get("label", "")
            
            # Check: Long BACKCHANNEL
            if label == "BACKCHANNEL" and len(text.split()) > 5:
                warnings.append(
                    f"{json_file.name}:{segment['id']} - Long BACKCHANNEL: '{text[:50]}...'"
                )
            
            # Check: Short YIELD/HOLD
            if label in ["YIELD", "HOLD"] and len(text.split()) <= 1:
                warnings.append(
                    f"{json_file.name}:{segment['id']} - Short {label}: '{text}'"
                )
            
            # Check: Has issues
            if segment.get("issues"):
                warnings.append(
                    f"{json_file.name}:{segment['id']} - Flagged issues: {segment['issues']}"
                )
    
    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Merge reviewed labels từ Label Studio"
    )
    
    parser.add_argument("--auto", "-a", required=True, help="Thư mục auto-labeled JSON")
    parser.add_argument("--reviewed", "-r", required=True, help="Label Studio export JSON")
    parser.add_argument("--output", "-o", default="data/processed/final", help="Output dir")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    
    args = parser.parse_args()
    
    # Merge
    stats = merge_labels(args.auto, args.reviewed, args.output)
    
    print(f"\n📊 Merge Statistics:")
    print(f"   Total segments: {stats['total_segments']}")
    print(f"   Reviewed: {stats['reviewed']} ({stats['reviewed']/stats['total_segments']*100:.1f}%)")
    print(f"   Auto only: {stats['auto_only']}")
    print(f"   Label changes: {stats['label_changes']}")
    print(f"   With issues: {stats['with_issues']}")
    print(f"\n   Final labels: {stats['final_labels']}")
    
    # Validate
    if args.validate:
        print(f"\n🔍 Validating...")
        warnings = validate_merged(args.output)
        
        if warnings:
            print(f"   ⚠️  Found {len(warnings)} warnings:")
            for w in warnings[:10]:  # Show first 10
                print(f"      - {w}")
            if len(warnings) > 10:
                print(f"      ... and {len(warnings) - 10} more")
        else:
            print("   ✅ No issues found!")
    
    print(f"\n✅ Merged data saved to {args.output}")


if __name__ == "__main__":
    main()
