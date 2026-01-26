#!/usr/bin/env python3
"""
02_llm_process.py - LLM-based Audio Processing with Gemini Multimodal

Thay thế WhisperX pipeline bằng Gemini 2.0 Flash để:
1. Transcribe audio (ASR)
2. Diarization (speaker identification)
3. Turn segmentation
4. Turn-taking labeling

Usage:
    python scripts/02_llm_process.py --input datasets/raw/youtube --output datasets/processed/llm
    python scripts/02_llm_process.py --input audio.wav --output output/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Encoding fix for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Google Gemini - New API
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Please install google-genai: pip install google-genai")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích hội thoại tiếng Việt. Nhiệm vụ của bạn:

1. **Transcribe**: Chuyển audio thành text chính xác
2. **Diarization**: Nhận diện người nói (SPEAKER_01, SPEAKER_02, ...)
3. **Segmentation**: Chia thành các lượt nói (turn) - mỗi turn chỉ có 1 người nói
4. **Labeling**: Gán nhãn turn-taking cho mỗi turn

## ĐỊNH NGHĨA TURN-TAKING LABELS:

- **YIELD**: Người nói kết thúc, sẵn sàng nhường lượt. Dấu hiệu: giọng xuống, câu hoàn chỉnh, có từ kết như "nhé", "ạ", "không?"
- **HOLD**: Người nói chưa xong, muốn giữ lượt. Dấu hiệu: giọng treo, câu chưa hoàn chỉnh, có từ nối "mà", "nhưng", "và"
- **BACKCHANNEL**: Tín hiệu lắng nghe ngắn như "ừ", "vâng", "à", "thế à" - không lấy lượt
- **COOPERATIVE_INTERRUPT**: Ngắt lời hợp tác - hoàn thành câu cho người khác hoặc hỗ trợ
- **COMPETITIVE_INTERRUPT**: Ngắt lời cạnh tranh - muốn lấy lượt nói

## QUY TẮC QUAN TRỌNG:

1. Mỗi turn CHỈ chứa lời của 1 người duy nhất
2. Khi có speaker change → BẮT BUỘC tạo turn mới
3. Xác định speaker dựa trên giọng nói, không phải nội dung
4. Estimate timestamps dạng "MM:SS" (phút:giây)
5. Trả về JSON hợp lệ, không có comment

## OUTPUT FORMAT:

```json
{
  "speakers_detected": 2,
  "speakers": {
    "SPEAKER_01": {"role": "host/guest", "gender": "male/female"},
    "SPEAKER_02": {"role": "host/guest", "gender": "male/female"}
  },
  "turns": [
    {
      "turn_id": 0,
      "speaker": "SPEAKER_01",
      "text": "Xin chào mọi người...",
      "start_time": "0:00",
      "end_time": "0:26",
      "turn_taking_label": "YIELD",
      "label_reason": "Câu chào hoàn chỉnh, giọng xuống",
      "confidence": 0.9
    }
  ]
}
```"""

USER_PROMPT_TEMPLATE = """Hãy phân tích file audio này và trả về JSON theo format đã định.

Lưu ý:
- Đây là podcast/interview tiếng Việt
- Có khoảng {expected_speakers} người nói
- Hãy tách CHÍNH XÁC từng lượt nói - mỗi turn chỉ có 1 speaker
- Ước lượng timestamps dạng "MM:SS"

Trả về ONLY JSON, không có text khác."""


# ==============================================================================
# GEMINI PROCESSOR (New API)
# ==============================================================================

class GeminiAudioProcessor:
    """Process audio files using Gemini Multimodal API (new google.genai)."""
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """Initialize processor."""
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google API key not found. "
                    "Set GOOGLE_API_KEY environment variable or pass via --api-key."
                )
        
        # New API: Create client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Valid labels for validation
        self.VALID_LABELS = {
            "YIELD", "HOLD", "BACKCHANNEL", 
            "COOPERATIVE_INTERRUPT", "COMPETITIVE_INTERRUPT"
        }
        
        print(f"✅ Initialized Gemini processor with model: {model}")
    
    def upload_audio(self, audio_path: str):
        """Upload audio file to Gemini."""
        print(f"   📤 Uploading audio: {Path(audio_path).name}")
        
        # Determine MIME type
        ext = Path(audio_path).suffix.lower()
        mime_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg"
        }
        mime_type = mime_types.get(ext, "audio/wav")
        
        # New API: Upload file
        audio_file = self.client.files.upload(
            file=audio_path,
            config={"mime_type": mime_type}
        )
        
        # Wait for processing
        while audio_file.state.name == "PROCESSING":
            print("   ⏳ Waiting for audio processing...")
            time.sleep(2)
            audio_file = self.client.files.get(name=audio_file.name)
        
        if audio_file.state.name == "FAILED":
            raise RuntimeError(f"Audio upload failed: {audio_file.state.name}")
        
        print(f"   ✅ Audio uploaded successfully")
        return audio_file
    
    def process_audio(
        self,
        audio_path: str,
        expected_speakers: int = 2
    ) -> Dict:
        """Process audio file with Gemini."""
        
        # Upload audio
        audio_file = self.upload_audio(audio_path)
        
        try:
            # Create prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                expected_speakers=expected_speakers
            )
            
            # Generate with retries
            for attempt in range(self.max_retries):
                try:
                    print(f"   🤖 Sending to Gemini (attempt {attempt + 1})...")
                    
                    # New API: Generate content
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_text(text=SYSTEM_PROMPT),
                                    types.Part.from_uri(
                                        file_uri=audio_file.uri,
                                        mime_type=audio_file.mime_type
                                    ),
                                    types.Part.from_text(text=user_prompt)
                                ]
                            )
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            top_p=0.95,
                            max_output_tokens=65536,  # Increase output limit
                        )
                    )
                    
                    # Parse response
                    result = self._parse_response(response.text)
                    print(f"   ✅ Successfully parsed {len(result.get('turns', []))} turns")
                    return result
                    
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ JSON parse error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
                        
                except Exception as e:
                    print(f"   ⚠️ Error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
        finally:
            # Clean up uploaded file
            try:
                self.client.files.delete(name=audio_file.name)
                print(f"   🗑️ Cleaned up uploaded file")
            except Exception:
                pass  # Ignore cleanup errors
    
    def _repair_truncated_json(self, text: str) -> str:
        """Attempt to repair truncated JSON by closing open structures."""
        import re

        # Find last complete turn object by looking for complete turn patterns
        # A complete turn ends with } and might be followed by , or be the last one

        # Strategy: find all complete turn objects and truncate after the last one
        # Look for pattern: complete JSON object with turn_id

        # First, try to find the last complete turn by finding balanced braces
        # Start from the turns array
        turns_match = re.search(r'"turns"\s*:\s*\[', text)
        if turns_match:
            turns_start = turns_match.end()

            # Find all complete objects in the turns array
            last_complete_end = turns_start
            brace_count = 0
            in_str = False
            escape = False
            obj_start = -1

            for i in range(turns_start, len(text)):
                char = text[i]

                if escape:
                    escape = False
                    continue
                if char == '\\' and in_str:
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue

                if char == '{':
                    if brace_count == 0:
                        obj_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and obj_start >= 0:
                        # Found a complete object
                        last_complete_end = i + 1
                        obj_start = -1

            if last_complete_end > turns_start:
                # Truncate after last complete turn
                text = text[:last_complete_end]

                # Remove trailing comma if present
                text = text.rstrip()
                if text.endswith(','):
                    text = text[:-1]

                # Close the turns array and main object
                text += '\n  ]\n}'
                return text

        # Fallback: simple repair
        # Remove any trailing incomplete content after last },
        last_complete = text.rfind('},')
        if last_complete > 0:
            text = text[:last_complete + 1]
        else:
            last_complete = text.rfind('}')
            if last_complete > 0:
                text = text[:last_complete + 1]

        # Remove trailing commas
        text = text.rstrip()
        if text.endswith(','):
            text = text[:-1]

        # Count and close open structures
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')

        text += ']' * open_brackets
        text += '}' * open_braces

        return text

    def _parse_response(self, response_text: str) -> Dict:
        """Parse and validate Gemini response."""

        # Clean response - remove markdown code blocks
        text = response_text.strip()

        # Remove ```json ... ``` wrapper
        if text.startswith("```"):
            # Find the end of first line
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove trailing ```
            if text.endswith("```"):
                text = text[:-3]

        text = text.strip()

        # Try to parse JSON, repair if truncated
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to repair truncated JSON
            print(f"   ⚠️ Attempting to repair truncated JSON...")
            repaired_text = self._repair_truncated_json(text)
            data = json.loads(repaired_text)
            print(f"   ✅ JSON repair successful")
        
        # Validate structure
        if "turns" not in data:
            raise ValueError("Response missing 'turns' field")
        
        # Validate and fix turns
        for i, turn in enumerate(data["turns"]):
            # Ensure required fields
            if "turn_id" not in turn:
                turn["turn_id"] = i
            if "speaker" not in turn:
                turn["speaker"] = "SPEAKER_01"
            if "text" not in turn:
                turn["text"] = ""
            if "turn_taking_label" not in turn:
                turn["turn_taking_label"] = "YIELD"
            
            # Validate label
            if turn["turn_taking_label"] not in self.VALID_LABELS:
                turn["turn_taking_label"] = "YIELD"
                turn["confidence"] = 0.5
            
            # Ensure confidence
            if "confidence" not in turn:
                turn["confidence"] = 0.8
        
        return data
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        expected_speakers: int = 2
    ) -> Dict:
        """Process a single audio file and save result."""
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        print(f"\n📂 Processing: {input_path.name}")
        
        # Process audio
        result = self.process_audio(str(input_path), expected_speakers)
        
        # Add metadata
        output_data = {
            "audio_file": input_path.name,
            "audio_path": str(input_path),
            "processing_method": self.model_name,
            "processing_date": datetime.now().isoformat(),
            "expected_speakers": expected_speakers,
            **result,
            "statistics": self._calculate_statistics(result)
        }
        
        # Add turn-taking events
        output_data["turn_taking_events"] = self._extract_events(result.get("turns", []))
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 Saved to: {output_path}")
        
        return output_data
    
    def _calculate_statistics(self, data: Dict) -> Dict:
        """Calculate statistics from processed data."""
        turns = data.get("turns", [])
        
        if not turns:
            return {}
        
        # Count speakers
        speaker_counts = {}
        label_counts = {}
        
        for turn in turns:
            speaker = turn.get("speaker", "UNKNOWN")
            label = turn.get("turn_taking_label", "UNKNOWN")
            
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "total_turns": len(turns),
            "speaker_counts": speaker_counts,
            "label_distribution": label_counts,
            "avg_confidence": sum(t.get("confidence", 0) for t in turns) / len(turns)
        }
    
    def _extract_events(self, turns: List[Dict]) -> List[Dict]:
        """Extract turn-taking events from turns."""
        events = []
        
        for i in range(len(turns) - 1):
            current = turns[i]
            next_turn = turns[i + 1]
            
            if current.get("speaker") != next_turn.get("speaker"):
                events.append({
                    "event_id": len(events),
                    "from_turn": current.get("turn_id", i),
                    "to_turn": next_turn.get("turn_id", i + 1),
                    "from_speaker": current.get("speaker"),
                    "to_speaker": next_turn.get("speaker"),
                    "from_label": current.get("turn_taking_label"),
                    "transition_type": self._classify_transition(
                        current.get("turn_taking_label", "YIELD")
                    )
                })
        
        return events
    
    def _classify_transition(self, label: str) -> str:
        """Classify transition type based on turn-taking label."""
        if label in ["COOPERATIVE_INTERRUPT", "COMPETITIVE_INTERRUPT"]:
            return "INTERRUPT"
        elif label == "BACKCHANNEL":
            return "BACKCHANNEL"
        elif label == "HOLD":
            return "INTERRUPTED"  # Speaker was holding but got interrupted
        else:
            return "SMOOTH"


# ==============================================================================
# MAIN
# ==============================================================================

def process_directory(
    input_dir: str,
    output_dir: str,
    processor: GeminiAudioProcessor,
    expected_speakers: int = 2
) -> Dict:
    """Process all audio files in a directory."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find audio files
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    audio_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        print(f"❌ No audio files found in {input_dir}")
        return {}
    
    print(f"📂 Found {len(audio_files)} audio files")
    
    results = {}
    success_count = 0
    error_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
        
        # Output path
        output_file = output_dir / f"{audio_file.stem}.json"
        
        # Skip if already processed
        if output_file.exists():
            print(f"   ⏭️ Already processed, skipping")
            continue
        
        try:
            result = processor.process_file(
                str(audio_file),
                str(output_file),
                expected_speakers
            )
            results[audio_file.name] = result
            success_count += 1
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            error_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Processing Complete")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Errors: {error_count}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based Audio Processing with Gemini Multimodal"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="datasets/processed/llm",
        help="Output directory"
    )
    parser.add_argument(
        "--api-key",
        help="Google API key (or set GOOGLE_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use"
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=2,
        help="Expected number of speakers"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries on error"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    try:
        processor = GeminiAudioProcessor(
            model=args.model,
            api_key=args.api_key,
            max_retries=args.retries
        )
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if input_path.is_file():
        # Single file
        output_file = output_dir / f"{input_path.stem}.json"
        processor.process_file(
            str(input_path),
            str(output_file),
            args.speakers
        )
    elif input_path.is_dir():
        # Directory
        process_directory(
            str(input_path),
            str(output_dir),
            processor,
            args.speakers
        )
    else:
        print(f"❌ Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
