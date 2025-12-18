"""
Proactive AI Friend - Response Classifier
メッセージに対してリアクション/返信/無視を判定する
"""

from openai import AsyncOpenAI
from typing import Optional
from dataclasses import dataclass
import json
import re

import config


@dataclass
class ResponseDecision:
    """応答の判定結果"""
    action: str  # "reply", "react", "none"
    reaction: Optional[str]  # リアクションの場合の絵文字
    reason: str  # 判定理由


# よく使うリアクション
REACTIONS = {
    "acknowledge": "👍",      # 了解系
    "thanks": "😊",           # ありがとう系
    "understood": "👌",       # わかった系
    "funny": "😂",            # 面白い系
    "sad": "🥲",              # 悲しい系
    "love": "❤️",             # 好き系
    "cool": "🔥",             # かっこいい/すごい系
    "thinking": "🤔",         # 考え中系
    "surprise": "😮",         # 驚き系
    "celebrate": "🎉",        # お祝い系
    "sleepy": "😴",           # 眠い系
    "food": "🤤",             # 美味しそう系
    "eyes": "👀",             # 見てる/気になる系
}


class ResponseClassifier:
    """
    メッセージに対する応答タイプを判定するクラス
    
    判定タイプ:
    - reply: 返信が必要（質問、相談、話題提供など）
    - react: リアクションで十分（了解、ありがとう、相槌など）
    - none: 何もしなくていい（独り言、誤送信っぽいなど）
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    
    async def classify(
        self, 
        message: str, 
        conversation_context: str = ""
    ) -> ResponseDecision:
        """
        メッセージの応答タイプを判定
        """
        prompt = f"""
あなたは友達とのチャットで、相手のメッセージにどう反応するか判断します。

## 相手のメッセージ
「{message}」

## 最近の会話の流れ
{conversation_context if conversation_context else "（なし）"}

## 判定基準

### reply（返信が必要）
- 質問されている
- 相談や悩みを話している
- 新しい話題を振ってきた
- 意見や感想を求めている
- 長めの文章で何かを伝えようとしている

### react（リアクションで十分）
- 「おk」「了解」「りょ」などの短い返事
- 「ありがとう」「さんきゅー」などのお礼
- 「わかった」「なるほど」などの相槌
- 「w」「草」「笑」などの笑い
- 写真や画像だけ
- 「おやすみ」「またね」などの挨拶の返事
- 前の会話の締めくくり的な発言

### none（何もしなくていい）
- 明らかな誤送信
- Botへの呼びかけではなさそう

## リアクションの種類
- acknowledge: 👍（了解、OK系）
- thanks: 😊（ありがとう系）
- understood: 👌（わかった、なるほど系）
- funny: 😂（面白い、笑い系）
- sad: 🥲（悲しい、残念系）
- love: ❤️（好き、嬉しい系）
- cool: 🔥（すごい、かっこいい系）
- thinking: 🤔（考え中、悩み系）
- surprise: 😮（驚き系）
- celebrate: 🎉（お祝い系）
- sleepy: 😴（眠い、疲れた系）
- food: 🤤（美味しそう系）
- eyes: 👀（気になる、見てる系）

## 出力形式（JSON）
{{
    "action": "reply" or "react" or "none",
    "reaction_type": "リアクションの種類（actionがreactの場合のみ）",
    "reason": "判定理由（1文）"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.choices[0].message.content
            result = self._extract_json(text)
            
            if not result:
                # パース失敗時はデフォルトで返信
                return ResponseDecision(
                    action="reply",
                    reaction=None,
                    reason="Parse error - defaulting to reply"
                )
            
            action = result.get("action", "reply")
            reaction_type = result.get("reaction_type")
            reaction = REACTIONS.get(reaction_type) if reaction_type else None
            
            return ResponseDecision(
                action=action,
                reaction=reaction,
                reason=result.get("reason", "")
            )
            
        except Exception as e:
            print(f"Classification error: {e}")
            # エラー時はデフォルトで返信
            return ResponseDecision(
                action="reply",
                reaction=None,
                reason=f"Error: {str(e)}"
            )
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """テキストからJSONを抽出"""
        # コードブロック内のJSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 直接JSON
        try:
            start_idx = text.find('{')
            if start_idx != -1:
                bracket_count = 0
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            return json.loads(text[start_idx:i+1])
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
