"""
LLM-as-a-Judge for automatic turn-taking label generation.
Uses Gemini API to classify Turn-Relevance Points in Vietnamese conversations.
"""

import json
from typing import List, Dict, Optional
import os

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ học về hội thoại tiếng Việt.

Nhiệm vụ: Phân tích đoạn hội thoại và xác định điểm chuyển lượt (Turn-Relevance Points).

Với mỗi câu/phát ngôn, hãy gán nhãn:
- YIELD: Người nói kết thúc, sẵn sàng để người khác nói (có hư từ: nhé, nhỉ, à, hả, ạ)
- HOLD: Người nói chưa xong, sẽ tiếp tục (có hư từ: mà, thì, là, nhưng mà, vì)  
- BACKCHANNEL: Phản hồi ngắn không chiếm lượt (ừ, vâng, ờ, à, thế à, vậy hả)

Trả về JSON array với format:
[
  {"text": "...", "speaker": "A/B", "label": "YIELD/HOLD/BACKCHANNEL", "confidence": 0.0-1.0, "reason": "..."}
]
"""


class LLMTurnLabeler:
    """Use LLM to automatically label turn-taking points."""
    
    # Vietnamese discourse markers
    YIELD_MARKERS = ['nhé', 'nhỉ', 'à', 'hả', 'ạ', 'không', 'chứ', 'hen', 'nha']
    HOLD_MARKERS = ['mà', 'thì', 'là', 'nhưng', 'vì', 'nên', 'nếu', 'khi', 'rồi']
    BACKCHANNEL = ['ừ', 'vâng', 'ờ', 'dạ', 'ừm', 'thế à', 'vậy hả', 'à']
    
    def __init__(
        self, 
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None
    ):
        if not HAS_GENAI:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model)
        
    def label_dialogue(
        self, 
        dialogue: List[Dict[str, str]],
        use_llm: bool = True
    ) -> List[Dict]:
        """
        Label a dialogue with turn-taking annotations.
        
        Args:
            dialogue: List of {"speaker": "A", "text": "...", "start": 0.0, "end": 1.5}
            use_llm: If True, use LLM. If False, use rule-based fallback.
        
        Returns:
            List of labeled utterances with turn-taking labels
        """
        if not use_llm:
            return self._fallback_labeling(dialogue)
        
        # Format dialogue for LLM
        formatted = "\n".join([
            f"[{u['speaker']}] ({u.get('start', 0):.1f}s - {u.get('end', 0):.1f}s): {u['text']}"
            for u in dialogue
        ])
        
        prompt = f"""{SYSTEM_PROMPT}

Phân tích đoạn hội thoại sau và gán nhãn turn-taking:

{formatted}

Trả về JSON array với nhãn cho từng phát ngôn."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            labels = json.loads(response.text)
            return self._merge_labels(dialogue, labels)
            
        except Exception as e:
            print(f"LLM labeling failed: {e}. Using fallback.")
            return self._fallback_labeling(dialogue)
    
    def _merge_labels(
        self, 
        dialogue: List[Dict], 
        labels: List[Dict]
    ) -> List[Dict]:
        """Merge LLM labels back into dialogue structure."""
        for i, utt in enumerate(dialogue):
            if i < len(labels):
                label = labels[i]
                utt['label'] = label.get('label', 'HOLD')
                utt['confidence'] = label.get('confidence', 0.5)
                utt['reason'] = label.get('reason', '')
            else:
                utt['label'] = 'YIELD'
                utt['confidence'] = 0.5
        return dialogue
    
    def _fallback_labeling(self, dialogue: List[Dict]) -> List[Dict]:
        """Rule-based fallback when LLM fails."""
        for utt in dialogue:
            text = utt['text'].lower().strip()
            words = text.split()
            
            # Check backchannel first (short responses)
            if len(words) <= 3 and any(bc in text for bc in self.BACKCHANNEL):
                utt['label'] = 'BACKCHANNEL'
                utt['confidence'] = 0.8
            # Check for yield markers at end
            elif words and any(words[-1].endswith(ym) for ym in self.YIELD_MARKERS):
                utt['label'] = 'YIELD'
                utt['confidence'] = 0.7
            # Check for hold markers
            elif any(hm in words for hm in self.HOLD_MARKERS):
                utt['label'] = 'HOLD'
                utt['confidence'] = 0.6
            else:
                # Default: assume yield at end of utterance
                utt['label'] = 'YIELD'
                utt['confidence'] = 0.5
            
            utt['reason'] = 'rule-based'
        
        return dialogue
    
    def label_batch(
        self,
        dialogues: List[List[Dict]],
        use_llm: bool = True
    ) -> List[List[Dict]]:
        """Label multiple dialogues."""
        results = []
        for i, dialogue in enumerate(dialogues):
            print(f"Labeling dialogue {i+1}/{len(dialogues)}")
            labeled = self.label_dialogue(dialogue, use_llm=use_llm)
            results.append(labeled)
        return results
