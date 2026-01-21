#!/usr/bin/env python3
"""
Script 03: LLM Pre-labeling với Gemini API

Gán nhãn YIELD/HOLD/BACKCHANNEL tự động cho các segments.

Yêu cầu:
    pip install google-generativeai
    export GOOGLE_API_KEY='your_api_key'

Usage:
    python scripts/03_llm_prelabel.py --input data/processed/auto --output data/processed/labeled
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

try:
    import google.generativeai as genai
except ImportError:
    print("❌ Cần cài đặt: pip install google-generativeai")
    sys.exit(1)


# Prompt template cho Gemini
SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ học về hội thoại tiếng Việt.

Nhiệm vụ: Phân tích đoạn hội thoại và gán nhãn TURN-TAKING cho MỖI phát ngôn.

3 LOẠI NHÃN:

1. YIELD (Nhường lời):
   - Người nói KẾT THÚC lượt, sẵn sàng để người khác nói
   - Dấu hiệu: Hư từ cuối câu (nhé, nhỉ, à, hả, ạ, hen, nha), giọng đi xuống, câu hỏi
   - Ví dụ: "Anh đi đâu đấy nhỉ?", "Em hiểu rồi ạ"

2. HOLD (Giữ lời):
   - Người nói CHƯA XONG, sẽ tiếp tục
   - Dấu hiệu: Câu còn dang dở, có "mà", "thì", "là", "vì", "nhưng", giọng treo
   - Ví dụ: "Tại vì hôm qua...", "Anh nghĩ là..."

3. BACKCHANNEL (Phản hồi ngắn):
   - Phản hồi ngắn KHÔNG chiếm lượt nói
   - Thường ≤3 từ, chỉ để thể hiện đang nghe
   - Ví dụ: "ừ", "vâng", "ờ", "à", "thế à", "vậy hả", "đúng rồi"

LƯU Ý QUAN TRỌNG:
- Nếu phát ngôn ngắn (<3 từ) và chỉ là phản hồi → BACKCHANNEL
- Nếu câu hỏi → thường là YIELD
- Nếu câu chưa hoàn chỉnh → HOLD"""

USER_PROMPT_TEMPLATE = """Phân tích hội thoại sau và gán nhãn cho TỪNG phát ngôn:

{conversation}

Trả về JSON array với format:
[
  {{"segment_id": 0, "label": "YIELD", "confidence": 0.9, "reason": "có 'nhỉ' cuối câu"}},
  ...
]

CHỈ TRẢ VỀ JSON, KHÔNG CÓ TEXT KHÁC."""


class LLMLabeler:
    """LLM-based turn-taking labeler sử dụng Gemini"""
    
    # Rule-based markers cho fallback và validation
    YIELD_MARKERS = ['nhé', 'nhỉ', 'à', 'hả', 'ạ', 'hen', 'nha', 'không', 'chứ', 'hả']
    HOLD_MARKERS = ['mà', 'thì', 'là', 'vì', 'nhưng', 'nên', 'nếu', 'khi', 'rồi']
    BACKCHANNEL_WORDS = ['ừ', 'vâng', 'ờ', 'dạ', 'ừm', 'ok', 'được']
    
    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Cần GOOGLE_API_KEY! Set: export GOOGLE_API_KEY='...'")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.generation_config = {
            "temperature": 0.1,  # Low temperature cho consistency
            "response_mime_type": "application/json"
        }
    
    def _format_conversation(self, segments: List[Dict]) -> str:
        """Format segments thành text cho LLM"""
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "?")
            text = seg.get("text", "")
            seg_id = seg.get("id", 0)
            lines.append(f"[{seg_id}] {speaker}: \"{text}\"")
        return "\n".join(lines)
    
    def _rule_based_label(self, text: str) -> tuple:
        """Fallback rule-based labeling"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Backchannel: ngắn và chứa marker
        if len(words) <= 3:
            if any(bc in text_lower for bc in self.BACKCHANNEL_WORDS):
                return "BACKCHANNEL", 0.8, "rule: short + backchannel word"
        
        # Yield: kết thúc bằng marker
        if words:
            last_word = words[-1]
            if any(last_word.endswith(ym) for ym in self.YIELD_MARKERS):
                return "YIELD", 0.7, f"rule: ends with '{last_word}'"
        
        # Backchannel: rất ngắn
        if len(words) <= 2:
            return "BACKCHANNEL", 0.6, "rule: very short utterance"
        
        # Hold: chứa hold marker ở giữa/cuối
        for hm in self.HOLD_MARKERS:
            if hm in words[-3:]:  # Trong 3 từ cuối
                return "HOLD", 0.6, f"rule: contains '{hm}'"
        
        # Default: yield (end of utterance)
        return "YIELD", 0.5, "rule: default"
    
    def label_segments(
        self, 
        segments: List[Dict],
        use_llm: bool = True,
        chunk_size: int = 20
    ) -> List[Dict]:
        """
        Gán nhãn cho các segments.
        
        Args:
            segments: List segments từ whisperX
            use_llm: Dùng LLM hay chỉ rule-based
            chunk_size: Số segments gửi mỗi lần (tránh token limit)
        """
        if not use_llm:
            # Rule-based only
            for seg in segments:
                label, conf, reason = self._rule_based_label(seg.get("text", ""))
                seg["auto_label"] = label
                seg["confidence"] = conf
                seg["label_reason"] = reason
            return segments
        
        # LLM labeling theo chunks
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            
            try:
                # Format conversation
                conv_text = self._format_conversation(chunk)
                prompt = USER_PROMPT_TEMPLATE.format(conversation=conv_text)
                
                # Call API
                response = self.model.generate_content(
                    [
                        {"role": "user", "parts": [SYSTEM_PROMPT]},
                        {"role": "model", "parts": ["Tôi hiểu. Tôi sẽ phân tích và gán nhãn YIELD/HOLD/BACKCHANNEL cho từng phát ngôn."]},
                        {"role": "user", "parts": [prompt]}
                    ],
                    generation_config=self.generation_config
                )
                
                # Parse response
                labels = json.loads(response.text)
                
                # Merge labels
                label_map = {l["segment_id"]: l for l in labels}
                for seg in chunk:
                    seg_id = seg["id"]
                    if seg_id in label_map:
                        seg["auto_label"] = label_map[seg_id]["label"]
                        seg["confidence"] = label_map[seg_id].get("confidence", 0.7)
                        seg["label_reason"] = label_map[seg_id].get("reason", "llm")
                    else:
                        # Fallback
                        label, conf, reason = self._rule_based_label(seg.get("text", ""))
                        seg["auto_label"] = label
                        seg["confidence"] = conf
                        seg["label_reason"] = reason
                
                # Rate limit
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ⚠️  LLM error: {e}. Using rule-based fallback.")
                for seg in chunk:
                    label, conf, reason = self._rule_based_label(seg.get("text", ""))
                    seg["auto_label"] = label
                    seg["confidence"] = conf
                    seg["label_reason"] = reason
        
        return segments
    
    def flag_for_review(self, segments: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """Đánh dấu segments cần human review"""
        for seg in segments:
            needs_review = False
            
            # Low confidence
            if seg.get("confidence", 1) < threshold:
                needs_review = True
            
            # Short text mà không phải backchannel
            text = seg.get("text", "")
            if len(text.split()) <= 2 and seg.get("auto_label") != "BACKCHANNEL":
                needs_review = True
            
            # Long text mà là backchannel
            if len(text.split()) > 5 and seg.get("auto_label") == "BACKCHANNEL":
                needs_review = True
            
            seg["needs_review"] = needs_review
        
        return segments


def process_file(input_path: str, output_path: str, labeler: LLMLabeler, use_llm: bool = True):
    """Process một file JSON"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    
    # Label
    segments = labeler.label_segments(segments, use_llm=use_llm)
    segments = labeler.flag_for_review(segments)
    
    # Stats
    stats = {
        "YIELD": sum(1 for s in segments if s.get("auto_label") == "YIELD"),
        "HOLD": sum(1 for s in segments if s.get("auto_label") == "HOLD"),
        "BACKCHANNEL": sum(1 for s in segments if s.get("auto_label") == "BACKCHANNEL"),
        "needs_review": sum(1 for s in segments if s.get("needs_review"))
    }
    
    data["segments"] = segments
    data["label_stats"] = stats
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="LLM Pre-labeling cho turn-taking segments"
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input dir/file")
    parser.add_argument("--output", "-o", default="data/processed/labeled", help="Output dir")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model")
    parser.add_argument("--api-key", help="Google API key (hoặc set GOOGLE_API_KEY)")
    parser.add_argument("--no-llm", action="store_true", help="Chỉ dùng rule-based")
    
    args = parser.parse_args()
    
    # Init labeler
    try:
        labeler = LLMLabeler(model=args.model, api_key=args.api_key)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        output_file = Path(args.output) / input_path.name
        print(f"📝 Processing: {input_path.name}")
        stats = process_file(str(input_path), str(output_file), labeler, not args.no_llm)
        print(f"   ✅ {stats}")
    
    elif input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
        print(f"📂 Found {len(json_files)} files")
        
        total_stats = {"YIELD": 0, "HOLD": 0, "BACKCHANNEL": 0, "needs_review": 0}
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] {json_file.name}")
            
            output_file = Path(args.output) / json_file.name
            if output_file.exists():
                print("   ⏭️  Already labeled, skipping")
                continue
            
            stats = process_file(str(json_file), str(output_file), labeler, not args.no_llm)
            print(f"   ✅ {stats}")
            
            for k, v in stats.items():
                total_stats[k] += v
        
        print(f"\n📊 Total: {total_stats}")
    
    else:
        print(f"❌ Invalid input: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
