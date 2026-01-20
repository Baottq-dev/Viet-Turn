#!/usr/bin/env python3
"""
Script 04: Export data cho Label Studio review

Tạo file JSON để import vào Label Studio cho human review.

Usage:
    python scripts/04_export_labelstudio.py --input data/processed/labeled --output data/labelstudio
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def create_labelstudio_task(
    segment: Dict,
    audio_file: str,
    source_file: str,
    audio_base_url: str = "/data/local-files/?d=audio"
) -> Dict:
    """
    Tạo một task cho Label Studio.
    
    Returns:
        Task dict theo Label Studio format
    """
    return {
        "data": {
            # Audio segment info
            "audio": f"{audio_base_url}/{audio_file}",
            "segment_start": segment.get("start", 0),
            "segment_end": segment.get("end", 0),
            
            # Text & speaker
            "text": segment.get("text", ""),
            "speaker": segment.get("speaker", "UNKNOWN"),
            
            # Auto labels
            "auto_label": segment.get("auto_label", "YIELD"),
            "confidence": round(segment.get("confidence", 0), 2),
            "label_reason": segment.get("label_reason", ""),
            
            # Metadata
            "segment_id": segment.get("id", 0),
            "source_file": source_file,
            "audio_file": audio_file
        },
        # Pre-fill với auto label
        "predictions": [{
            "model_version": "auto_v1",
            "result": [{
                "from_name": "turn_label",
                "to_name": "audio",
                "type": "choices",
                "value": {
                    "choices": [segment.get("auto_label", "YIELD")]
                }
            }]
        }]
    }


def export_for_labelstudio(
    input_dir: str,
    output_dir: str,
    audio_src_dir: str = None,
    review_only: bool = True,
    confidence_threshold: float = 0.7
) -> Dict:
    """
    Export tất cả segments sang Label Studio format.
    
    Args:
        input_dir: Thư mục chứa labeled JSON files
        output_dir: Thư mục output
        audio_src_dir: Thư mục chứa audio gốc (sẽ copy sang output)
        review_only: Chỉ export segments cần review (needs_review=True hoặc confidence thấp)
        confidence_threshold: Ngưỡng confidence để cần review
    
    Returns:
        Stats dict
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    stats = {"total": 0, "exported": 0, "by_label": {}}
    
    # Process each JSON file
    for json_file in sorted(input_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        audio_file = data.get("audio_file", json_file.stem + ".wav")
        source_file = json_file.name
        
        for segment in data.get("segments", []):
            stats["total"] += 1
            
            # Filter: chỉ export segments cần review
            if review_only:
                needs_review = segment.get("needs_review", False)
                low_confidence = segment.get("confidence", 1) < confidence_threshold
                
                if not (needs_review or low_confidence):
                    continue
            
            # Create task
            task = create_labelstudio_task(segment, audio_file, source_file)
            tasks.append(task)
            stats["exported"] += 1
            
            # Count by label
            label = segment.get("auto_label", "UNKNOWN")
            stats["by_label"][label] = stats["by_label"].get(label, 0) + 1
    
    # Save tasks
    tasks_file = output_path / "tasks.json"
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Exported {stats['exported']}/{stats['total']} segments to {tasks_file}")
    
    # Copy audio files nếu có
    if audio_src_dir:
        audio_output = output_path / "audio"
        audio_output.mkdir(exist_ok=True)
        
        audio_files = set(t["data"]["audio_file"] for t in tasks)
        copied = 0
        
        for audio_name in audio_files:
            src = Path(audio_src_dir) / audio_name
            dst = audio_output / audio_name
            
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        
        print(f"📁 Copied {copied} audio files to {audio_output}")
    
    # Create Label Studio config
    config = create_labeling_config()
    config_file = output_path / "labeling_config.xml"
    with open(config_file, "w") as f:
        f.write(config)
    
    print(f"📋 Created labeling config: {config_file}")
    
    # Create instructions
    instructions = create_review_instructions()
    instructions_file = output_path / "INSTRUCTIONS.md"
    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write(instructions)
    
    return stats


def create_labeling_config() -> str:
    """Tạo Label Studio XML config"""
    return """<View>
  <Header value="🎵 Audio Segment"/>
  <View style="display: flex; gap: 10px; margin-bottom: 10px;">
    <View style="flex: 1; padding: 10px; background: #f0f0f0; border-radius: 5px;">
      <Text name="time_info" value="⏱️ $segment_start s - $segment_end s"/>
    </View>
    <View style="flex: 1; padding: 10px; background: #e8f4e8; border-radius: 5px;">
      <Text name="speaker_info" value="👤 Speaker: $speaker"/>
    </View>
  </View>
  
  <Audio name="audio" value="$audio" hotkey="space"/>
  
  <Header value="📝 Transcript"/>
  <View style="padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">
    <Text name="transcript" value="$text" style="font-size: 16px;"/>
  </View>
  
  <Header value="🤖 Auto Label (confidence: $confidence)"/>
  <View style="padding: 5px 10px; background: #ffe0b2; border-radius: 3px; display: inline-block; margin-bottom: 10px;">
    <Text name="auto" value="$auto_label - $label_reason"/>
  </View>
  
  <Header value="✅ Your Label"/>
  <Choices name="turn_label" toName="audio" choice="single" showInline="true">
    <Choice value="YIELD" style="background: #c8e6c9;" hotkey="1"/>
    <Choice value="HOLD" style="background: #bbdefb;" hotkey="2"/>
    <Choice value="BACKCHANNEL" style="background: #fff9c4;" hotkey="3"/>
  </Choices>
  
  <Header value="⚠️ Issues (optional)"/>
  <Choices name="issues" toName="audio" choice="multiple" showInline="true">
    <Choice value="WRONG_SPEAKER"/>
    <Choice value="WRONG_TEXT"/>
    <Choice value="OVERLAP"/>
    <Choice value="NOISE"/>
    <Choice value="UNCLEAR"/>
  </Choices>
  
  <Header value="📝 Notes (optional)"/>
  <TextArea name="notes" toName="audio" rows="2" maxSubmissions="1"/>
</View>"""


def create_review_instructions() -> str:
    """Tạo hướng dẫn review"""
    return """# 📋 Hướng dẫn Review Turn-Taking Labels

## Cách sử dụng Label Studio

1. **Import data**: Settings → Import → Upload `tasks.json`
2. **Setup labeling**: Settings → Labeling Interface → Code → Paste nội dung từ `labeling_config.xml`
3. **Start labeling**: Click "Label All Tasks"

## Hotkeys

- `Space`: Play/Pause audio
- `1`: YIELD
- `2`: HOLD  
- `3`: BACKCHANNEL
- `Ctrl+Enter`: Submit & next

## Định nghĩa Labels

### YIELD (Nhường lời) - Phím 1
Người nói **KẾT THÚC** lượt, sẵn sàng để người khác nói.

**Dấu hiệu:**
- Hư từ cuối câu: "nhé", "nhỉ", "à", "hả", "ạ", "hen"
- Giọng đi xuống
- Câu hỏi

**Ví dụ:**
- "Anh đi đâu đấy **nhỉ**?"
- "Em hiểu rồi **ạ**"
- "Thế thì được rồi"

### HOLD (Giữ lời) - Phím 2
Người nói **CHƯA XONG**, sẽ tiếp tục.

**Dấu hiệu:**
- Câu còn dang dở
- Có "mà", "thì", "là", "vì", "nhưng"
- Giọng treo (không đi xuống)

**Ví dụ:**
- "Tại vì hôm qua..."
- "Anh nghĩ là..."
- "Cái này thì..."

### BACKCHANNEL (Phản hồi ngắn) - Phím 3
Phản hồi ngắn **KHÔNG chiếm lượt nói**.

**Dấu hiệu:**
- Thường ≤3 từ
- Chỉ để thể hiện đang nghe
- Không có nội dung mới

**Ví dụ:**
- "ừ", "vâng", "ờ", "à"
- "thế à", "vậy hả"
- "đúng rồi", "được"

## Checklist khi Review

1. ☐ **Nghe audio** trước khi đọc text
2. ☐ **Kiểm tra speaker** có đúng không
3. ☐ **Kiểm tra text** có đúng không (đặc biệt hư từ cuối)
4. ☐ **Chọn label** phù hợp
5. ☐ **Flag issues** nếu có vấn đề

## Khi nào flag Issues?

- `WRONG_SPEAKER`: Speaker bị gán sai
- `WRONG_TEXT`: Text sai so với audio
- `OVERLAP`: 2 người nói chồng lên nhau
- `NOISE`: Quá nhiều noise
- `UNCLEAR`: Không nghe rõ
"""


def main():
    parser = argparse.ArgumentParser(
        description="Export data sang Label Studio format"
    )
    
    parser.add_argument("--input", "-i", required=True, help="Thư mục labeled JSON")
    parser.add_argument("--output", "-o", default="data/labelstudio", help="Output dir")
    parser.add_argument("--audio-src", help="Thư mục audio gốc (để copy)")
    parser.add_argument("--all", action="store_true", help="Export tất cả, không chỉ needs_review")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    
    args = parser.parse_args()
    
    stats = export_for_labelstudio(
        args.input,
        args.output,
        audio_src_dir=args.audio_src,
        review_only=not args.all,
        confidence_threshold=args.threshold
    )
    
    print(f"\n📊 Stats: {stats}")
    print(f"\n🚀 Next steps:")
    print(f"   1. Start Label Studio: label-studio start")
    print(f"   2. Create project, import {args.output}/tasks.json")
    print(f"   3. Setup labeling interface with {args.output}/labeling_config.xml")


if __name__ == "__main__":
    main()
