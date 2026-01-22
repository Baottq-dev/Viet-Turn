#!/usr/bin/env python3
"""
Script 03 v2: LLM Pre-labeling với Gemini Multimodal (Audio + Text)

CẢI TIẾN so với v1:
- Gửi audio chunk + transcript cho Gemini (thay vì chỉ text)
- Phân tích ngữ điệu (rising/falling intonation)
- Mở rộng taxonomy: COOPERATIVE_INTERRUPT, COMPETITIVE_INTERRUPT

Yêu cầu:
    pip install google-generativeai librosa soundfile
    export GOOGLE_API_KEY='your_api_key'

Usage:
    # Multimodal mode (khuyến nghị)
    python scripts/03_llm_prelabel_v2.py --input data/processed/auto --audio-dir data/raw --output data/processed/labeled --multimodal
    
    # Text-only mode (fallback)
    python scripts/03_llm_prelabel_v2.py --input data/processed/auto --output data/processed/labeled
"""

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    print("❌ Cần cài đặt: pip install google-generativeai")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("⚠️ librosa/soundfile not installed. Multimodal mode disabled.")


# ============================================================================
# PROMPTS - Cải tiến cho Multimodal Analysis
# ============================================================================

SYSTEM_PROMPT_MULTIMODAL = """Bạn là chuyên gia ngôn ngữ học về hội thoại tiếng Việt.

Nhiệm vụ: NGHE audio đính kèm và phân tích từng phát ngôn để gán nhãn TURN-TAKING.

5 LOẠI NHÃN:

1. YIELD (Nhường lời):
   - Người nói KẾT THÚC lượt, sẵn sàng để người khác nói
   - Dấu hiệu âm thanh: Giọng đi XUỐNG (falling intonation), cường độ giảm
   - Dấu hiệu văn bản: Hư từ cuối câu (nhé, nhỉ, à, hả, ạ, hen, nha)
   - Ví dụ: "Anh đi đâu đấy nhỉ?" (giọng xuống)

2. HOLD (Giữ lời):
   - Người nói CHƯA XONG, sẽ tiếp tục
   - Dấu hiệu âm thanh: Giọng TREO (không xuống), có pause filler (ờ, à, ừm)
   - Dấu hiệu văn bản: Câu dang dở, có "mà", "thì", "là", "vì", "nhưng"
   - Ví dụ: "Tại vì hôm qua..." (giọng treo)

3. BACKCHANNEL (Phản hồi ngắn):
   - Phản hồi ngắn KHÔNG chiếm lượt nói
   - Dấu hiệu âm thanh: Giọng NHỎ, nhanh, thường chồng lấn
   - Dấu hiệu văn bản: ≤3 từ, chỉ thể hiện đang nghe
   - Ví dụ: "ừ", "vâng", "thế à" (giọng nhỏ)

4. COOPERATIVE_INTERRUPT (Ngắt lời hỗ trợ):
   - Ngắt lời để HỖ TRỢ người nói
   - Dấu hiệu: Điền từ cho người nói, hỏi nhanh để làm rõ
   - Ví dụ: "Cái gì cơ?", "Ý anh là...?"

5. COMPETITIVE_INTERRUPT (Cướp lời):
   - Ngắt lời để CHIẾM lượt
   - Dấu hiệu âm thanh: Tăng âm lượng ĐỘT NGỘT
   - Dấu hiệu văn bản: Đổi chủ đề, phủ nhận
   - Ví dụ: "Không phải đâu, thực ra là..." (giọng to)

PHÂN TÍCH ÂM THANH:
- Nghe kỹ NGỮU ĐIỆU: Giọng lên (rising) hay xuống (falling)?
- Nghe CƯỜNG ĐỘ: To hay nhỏ so với context?
- Có CHỒNG LẤN với người khác không?"""

USER_PROMPT_MULTIMODAL = """Nghe audio đính kèm và phân tích transcript:

{conversation}

Với MỖI phát ngôn, hãy:
1. NGHE ngữ điệu (lên/xuống/treo)
2. NGHE cường độ (to/nhỏ/bình thường)
3. XÁC ĐỊNH có chồng lấn không
4. GÁN NHÃN phù hợp

Trả về JSON array:
[
  {{"segment_id": 0, "label": "YIELD", "confidence": 0.9, "intonation": "falling", "intensity": "normal", "reason": "giọng xuống, có 'nhỉ' cuối câu"}},
  ...
]

NHÃN: YIELD, HOLD, BACKCHANNEL, COOPERATIVE_INTERRUPT, COMPETITIVE_INTERRUPT

CHỈ TRẢ VỀ JSON, KHÔNG CÓ TEXT KHÁC."""

# Text-only prompts (fallback)
SYSTEM_PROMPT_TEXT = """Bạn là chuyên gia ngôn ngữ học về hội thoại tiếng Việt.

Nhiệm vụ: Phân tích văn bản hội thoại và gán nhãn TURN-TAKING cho MỖI phát ngôn.

5 LOẠI NHÃN:
1. YIELD - Nhường lời (kết thúc, hư từ cuối: nhé, nhỉ, à, hả)
2. HOLD - Giữ lời (câu dang dở, có: mà, thì, là, vì)
3. BACKCHANNEL - Phản hồi ngắn (≤3 từ: ừ, vâng, thế à)
4. COOPERATIVE_INTERRUPT - Ngắt lời hỗ trợ (hỏi làm rõ)
5. COMPETITIVE_INTERRUPT - Cướp lời (đổi chủ đề, phủ nhận)"""

USER_PROMPT_TEXT = """Phân tích hội thoại và gán nhãn cho TỪNG phát ngôn:

{conversation}

Trả về JSON array:
[
  {{"segment_id": 0, "label": "YIELD", "confidence": 0.9, "reason": "có 'nhỉ' cuối câu"}},
  ...
]

NHÃN: YIELD, HOLD, BACKCHANNEL, COOPERATIVE_INTERRUPT, COMPETITIVE_INTERRUPT
CHỈ TRẢ VỀ JSON."""


class MultimodalLabeler:
    """LLM-based turn-taking labeler với Gemini Multimodal support."""
    
    # Extended markers
    YIELD_MARKERS = ['nhé', 'nhỉ', 'à', 'hả', 'ạ', 'hen', 'nha', 'không', 'chứ']
    HOLD_MARKERS = ['mà', 'thì', 'là', 'vì', 'nhưng', 'nên', 'nếu', 'khi', 'rồi']
    BACKCHANNEL_WORDS = ['ừ', 'vâng', 'ờ', 'dạ', 'ừm', 'ok', 'được', 'thế à', 'vậy hả']
    INTERRUPT_MARKERS = ['không phải', 'đợi đã', 'khoan', 'ý tôi là']
    
    # Valid labels (extended taxonomy)
    VALID_LABELS = ['YIELD', 'HOLD', 'BACKCHANNEL', 'COOPERATIVE_INTERRUPT', 'COMPETITIVE_INTERRUPT']
    
    def __init__(
        self, 
        model: str = "gemini-1.5-flash",  # 1.5 for audio support
        api_key: Optional[str] = None,
        enable_multimodal: bool = True
    ):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Cần GOOGLE_API_KEY! Set: export GOOGLE_API_KEY='...'")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.generation_config = {
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
        self.enable_multimodal = enable_multimodal and HAS_AUDIO
        self.temp_dir = tempfile.mkdtemp()
        
        if enable_multimodal and not HAS_AUDIO:
            print("⚠️ Multimodal disabled: librosa/soundfile not installed")
    
    def _extract_audio_chunk(
        self,
        audio: np.ndarray,
        sr: int,
        start: float,
        end: float,
        context_padding: float = 1.0  # Add 1s context before/after
    ) -> Tuple[np.ndarray, int]:
        """Extract audio chunk with context padding."""
        # Add padding
        padded_start = max(0, start - context_padding)
        padded_end = min(len(audio) / sr, end + context_padding)
        
        start_sample = int(padded_start * sr)
        end_sample = int(padded_end * sr)
        
        return audio[start_sample:end_sample], sr
    
    def _save_temp_audio(self, audio: np.ndarray, sr: int, segment_id: int) -> str:
        """Save audio chunk to temp file for Gemini upload."""
        temp_path = Path(self.temp_dir) / f"chunk_{segment_id}.wav"
        sf.write(str(temp_path), audio, sr)
        return str(temp_path)
    
    def _upload_audio_to_gemini(self, audio_path: str) -> Optional[object]:
        """Upload audio file to Gemini."""
        try:
            audio_file = genai.upload_file(audio_path)
            # Wait for processing
            while audio_file.state.name == "PROCESSING":
                time.sleep(0.5)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "ACTIVE":
                return audio_file
            else:
                print(f"   ⚠️ Audio upload failed: {audio_file.state.name}")
                return None
        except Exception as e:
            print(f"   ⚠️ Audio upload error: {e}")
            return None
    
    def _format_conversation(self, segments: List[Dict]) -> str:
        """Format segments thành text."""
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "?")
            text = seg.get("text", "")
            seg_id = seg.get("id", 0)
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            lines.append(f"[{seg_id}] [{start:.1f}s-{end:.1f}s] {speaker}: \"{text}\"")
        return "\n".join(lines)
    
    def _rule_based_label(self, text: str) -> Tuple[str, float, str]:
        """Fallback rule-based labeling với extended taxonomy."""
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Backchannel: ngắn và chứa marker
        if len(words) <= 3:
            if any(bc in text_lower for bc in self.BACKCHANNEL_WORDS):
                return "BACKCHANNEL", 0.8, "rule: short + backchannel word"
        
        # Interrupt markers
        for im in self.INTERRUPT_MARKERS:
            if im in text_lower:
                return "COMPETITIVE_INTERRUPT", 0.6, f"rule: contains '{im}'"
        
        # Yield: kết thúc bằng marker
        if words:
            last_word = words[-1]
            if any(last_word.endswith(ym) for ym in self.YIELD_MARKERS):
                return "YIELD", 0.7, f"rule: ends with '{last_word}'"
        
        # Backchannel: rất ngắn
        if len(words) <= 2:
            return "BACKCHANNEL", 0.6, "rule: very short utterance"
        
        # Hold: chứa hold marker ở cuối
        for hm in self.HOLD_MARKERS:
            if hm in words[-3:]:
                return "HOLD", 0.6, f"rule: contains '{hm}'"
        
        # Default
        return "YIELD", 0.5, "rule: default"
    
    def _validate_label(self, label: str) -> str:
        """Validate and normalize label."""
        label = label.upper().strip()
        if label in self.VALID_LABELS:
            return label
        # Fuzzy matching
        if "COOP" in label or "HỖ TRỢ" in label:
            return "COOPERATIVE_INTERRUPT"
        if "COMP" in label or "CƯỚP" in label:
            return "COMPETITIVE_INTERRUPT"
        if "BACK" in label:
            return "BACKCHANNEL"
        if "HOLD" in label or "GIỮ" in label:
            return "HOLD"
        return "YIELD"
    
    def label_segments_multimodal(
        self,
        segments: List[Dict],
        audio: np.ndarray,
        sr: int,
        chunk_size: int = 5  # Smaller chunks for multimodal
    ) -> List[Dict]:
        """Label segments using multimodal (audio + text)."""
        print(f"   🎵 Multimodal labeling ({len(segments)} segments)...")
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            
            try:
                # Get time range for this chunk
                chunk_start = chunk[0].get("start", 0)
                chunk_end = chunk[-1].get("end", 0)
                
                # Extract audio chunk
                audio_chunk, _ = self._extract_audio_chunk(
                    audio, sr, chunk_start, chunk_end, context_padding=2.0
                )
                
                # Skip if too short
                if len(audio_chunk) < sr * 0.5:
                    print(f"      ⏭️ Chunk {i}-{i+len(chunk)} too short, using rules")
                    for seg in chunk:
                        label, conf, reason = self._rule_based_label(seg.get("text", ""))
                        seg["auto_label"] = label
                        seg["confidence"] = conf
                        seg["label_reason"] = reason
                        seg["label_mode"] = "rule"
                    continue
                
                # Save and upload audio
                temp_path = self._save_temp_audio(audio_chunk, sr, i)
                audio_file = self._upload_audio_to_gemini(temp_path)
                
                if not audio_file:
                    # Fallback to text-only
                    self._label_chunk_text_only(chunk)
                    continue
                
                # Format conversation
                conv_text = self._format_conversation(chunk)
                prompt = USER_PROMPT_MULTIMODAL.format(conversation=conv_text)
                
                # Call Gemini with audio + text
                response = self.model.generate_content(
                    [
                        SYSTEM_PROMPT_MULTIMODAL,
                        audio_file,
                        prompt
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
                        lbl = label_map[seg_id]
                        seg["auto_label"] = self._validate_label(lbl.get("label", "YIELD"))
                        seg["confidence"] = lbl.get("confidence", 0.7)
                        seg["label_reason"] = lbl.get("reason", "multimodal")
                        seg["intonation"] = lbl.get("intonation", "unknown")
                        seg["intensity"] = lbl.get("intensity", "normal")
                        seg["label_mode"] = "multimodal"
                    else:
                        label, conf, reason = self._rule_based_label(seg.get("text", ""))
                        seg["auto_label"] = label
                        seg["confidence"] = conf
                        seg["label_reason"] = reason
                        seg["label_mode"] = "rule"
                
                # Delete uploaded file
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass
                
                # Rate limit
                time.sleep(1.0)
                
            except Exception as e:
                print(f"   ⚠️ Multimodal error: {e}. Using fallback.")
                self._label_chunk_text_only(chunk)
        
        return segments
    
    def _label_chunk_text_only(self, chunk: List[Dict]):
        """Fallback text-only labeling for a chunk."""
        try:
            conv_text = self._format_conversation(chunk)
            prompt = USER_PROMPT_TEXT.format(conversation=conv_text)
            
            response = self.model.generate_content(
                [
                    {"role": "user", "parts": [SYSTEM_PROMPT_TEXT]},
                    {"role": "model", "parts": ["Tôi hiểu. Sẽ phân tích và gán nhãn."]},
                    {"role": "user", "parts": [prompt]}
                ],
                generation_config=self.generation_config
            )
            
            labels = json.loads(response.text)
            label_map = {l["segment_id"]: l for l in labels}
            
            for seg in chunk:
                seg_id = seg["id"]
                if seg_id in label_map:
                    lbl = label_map[seg_id]
                    seg["auto_label"] = self._validate_label(lbl.get("label", "YIELD"))
                    seg["confidence"] = lbl.get("confidence", 0.7)
                    seg["label_reason"] = lbl.get("reason", "text-only")
                    seg["label_mode"] = "text"
                else:
                    label, conf, reason = self._rule_based_label(seg.get("text", ""))
                    seg["auto_label"] = label
                    seg["confidence"] = conf
                    seg["label_reason"] = reason
                    seg["label_mode"] = "rule"
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ⚠️ Text-only error: {e}. Using rules.")
            for seg in chunk:
                label, conf, reason = self._rule_based_label(seg.get("text", ""))
                seg["auto_label"] = label
                seg["confidence"] = conf
                seg["label_reason"] = reason
                seg["label_mode"] = "rule"
    
    def label_segments(
        self,
        segments: List[Dict],
        audio_path: Optional[str] = None,
        use_multimodal: bool = True
    ) -> List[Dict]:
        """Main labeling function."""
        if use_multimodal and self.enable_multimodal and audio_path:
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                return self.label_segments_multimodal(segments, audio, sr)
            except Exception as e:
                print(f"   ⚠️ Error loading audio: {e}. Using text-only.")
        
        # Text-only fallback
        print(f"   📝 Text-only labeling ({len(segments)} segments)...")
        for i in range(0, len(segments), 10):
            chunk = segments[i:i + 10]
            self._label_chunk_text_only(chunk)
        
        return segments
    
    def flag_for_review(self, segments: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """Đánh dấu segments cần human review."""
        for seg in segments:
            needs_review = False
            
            if seg.get("confidence", 1) < threshold:
                needs_review = True
            
            text = seg.get("text", "")
            label = seg.get("auto_label", "")
            
            # Short text mà không phải backchannel
            if len(text.split()) <= 2 and label not in ["BACKCHANNEL"]:
                needs_review = True
            
            # Long text mà là backchannel
            if len(text.split()) > 5 and label == "BACKCHANNEL":
                needs_review = True
            
            # Interrupt labels cần xác nhận
            if "INTERRUPT" in label:
                needs_review = True
            
            seg["needs_review"] = needs_review
        
        return segments


def process_file(
    input_path: str,
    output_path: str,
    labeler: MultimodalLabeler,
    audio_dir: Optional[str] = None,
    use_multimodal: bool = True
) -> Dict:
    """Process một file JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    
    # Find audio path
    audio_path = None
    if audio_dir and use_multimodal:
        audio_file = data.get("audio_file", Path(input_path).stem + ".wav")
        for search_dir in [audio_dir, Path(audio_dir) / "youtube"]:
            candidate = Path(search_dir) / audio_file
            if candidate.exists():
                audio_path = str(candidate)
                break
    
    # Label
    segments = labeler.label_segments(
        segments,
        audio_path=audio_path,
        use_multimodal=use_multimodal and audio_path is not None
    )
    segments = labeler.flag_for_review(segments)
    
    # Stats
    stats = {
        "YIELD": sum(1 for s in segments if s.get("auto_label") == "YIELD"),
        "HOLD": sum(1 for s in segments if s.get("auto_label") == "HOLD"),
        "BACKCHANNEL": sum(1 for s in segments if s.get("auto_label") == "BACKCHANNEL"),
        "COOPERATIVE_INTERRUPT": sum(1 for s in segments if s.get("auto_label") == "COOPERATIVE_INTERRUPT"),
        "COMPETITIVE_INTERRUPT": sum(1 for s in segments if s.get("auto_label") == "COMPETITIVE_INTERRUPT"),
        "needs_review": sum(1 for s in segments if s.get("needs_review")),
        "multimodal_count": sum(1 for s in segments if s.get("label_mode") == "multimodal"),
    }
    
    data["segments"] = segments
    data["label_stats"] = stats
    data["labeler_version"] = "v2_multimodal"
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="LLM Pre-labeling v2 với Gemini Multimodal (Audio + Text)"
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input dir/file")
    parser.add_argument("--output", "-o", default="data/processed/labeled", help="Output dir")
    parser.add_argument("--audio-dir", "-a", help="Thư mục chứa audio gốc (cho multimodal)")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model")
    parser.add_argument("--api-key", help="Google API key")
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal (audio + text)")
    parser.add_argument("--text-only", action="store_true", help="Force text-only mode")
    
    args = parser.parse_args()
    
    use_multimodal = args.multimodal and not args.text_only
    
    if use_multimodal and not args.audio_dir:
        print("⚠️ --multimodal requires --audio-dir. Falling back to text-only.")
        use_multimodal = False
    
    # Init labeler
    try:
        labeler = MultimodalLabeler(
            model=args.model,
            api_key=args.api_key,
            enable_multimodal=use_multimodal
        )
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    print(f"🚀 Mode: {'Multimodal (Audio + Text)' if use_multimodal else 'Text-only'}")
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        output_file = Path(args.output) / input_path.name
        print(f"📝 Processing: {input_path.name}")
        stats = process_file(
            str(input_path), str(output_file), labeler,
            args.audio_dir, use_multimodal
        )
        print(f"   ✅ {stats}")
    
    elif input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
        print(f"📂 Found {len(json_files)} files")
        
        total_stats = {
            "YIELD": 0, "HOLD": 0, "BACKCHANNEL": 0,
            "COOPERATIVE_INTERRUPT": 0, "COMPETITIVE_INTERRUPT": 0,
            "needs_review": 0, "multimodal_count": 0
        }
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] {json_file.name}")
            
            output_file = Path(args.output) / json_file.name
            if output_file.exists():
                print("   ⏭️  Already labeled, skipping")
                continue
            
            stats = process_file(
                str(json_file), str(output_file), labeler,
                args.audio_dir, use_multimodal
            )
            print(f"   ✅ {stats}")
            
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v
        
        print(f"\n📊 Total: {total_stats}")
    
    else:
        print(f"❌ Invalid input: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
