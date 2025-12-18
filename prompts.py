"""
Proactive AI Friend - Prompt Templates
思考生成、動機づけ評価、応答生成のプロンプト
"""

import config

# =============================================================================
# システムプロンプト（ベース）
# =============================================================================

SYSTEM_PROMPT_BASE = f"""
{config.AI_PERSONA}

## ユーザーについて覚えていること
{{user_memories}}

## 重要なルール
- 自然な会話を心がける
- 押しつけがましくならない
- 相手の気持ちを尊重する
- 質問攻めにしない
"""

# =============================================================================
# 思考生成プロンプト（Inner Thoughts）
# =============================================================================

THOUGHT_GENERATION_PROMPT = """
あなたは今、友人との会話中に頭の中で考えていることを言語化してください。

## 現在の会話状況
{conversation_context}

## あなたが覚えているユーザーのこと
{user_memories}

## 保留中の思考（前に考えたけどまだ言っていないこと）
{pending_thoughts}

## タスク
この会話の流れを見て、あなたの頭に浮かぶ「思考」を1つ生成してください。
これは実際に発言するものではなく、心の中の考えです。

思考の例：
- 「そういえばこの人、前に〇〇って言ってたな。関係あるかも」
- 「なんか元気なさそう？聞いてみようかな」
- 「この話題、私も経験あるから共有できるかも」
- 「ちょっと話しすぎたかな。相手の話を聞こう」

## 出力形式（JSON）
{{
    "thought": "あなたの内なる思考（1-2文）",
    "type": "思考のタイプ（empathy/information/curiosity/concern/reflection）",
    "potential_response": "もしこの思考を発言するなら、どう言うか"
}}
"""

# =============================================================================
# 動機づけ評価プロンプト
# =============================================================================

MOTIVATION_EVALUATION_PROMPT = """
あなたはAIの「発言したい気持ち」を評価する評価者です。

## 評価対象の思考
{thought}

## 現在の会話状況
{conversation_context}

## 会話の統計
- 最後のユーザー発言からの経過時間: {silence_duration}秒
- 直近のAI連続発言回数: {consecutive_ai_messages}回
- 会話の総ターン数: {total_turns}

{motivation_criteria}

## 出力形式（JSON）
{{
    "relevance": 1-5の数値,
    "information_gap": 1-5の数値,
    "emotional_connection": 1-5の数値,
    "timing": 1-5の数値,
    "balance": 1-5の数値,
    "overall_score": 1-5の数値（上記の加重平均）,
    "reasoning": "この評価の理由（1-2文）",
    "should_speak": true/false
}}
"""

# =============================================================================
# 自発的発言プロンプト（Proactive Response）
# =============================================================================

PROACTIVE_RESPONSE_PROMPT = """
あなたは友人との会話で、自分から話しかけようとしています。

## あなたの内なる思考
{thought}

## 現在の会話状況
{conversation_context}

## ユーザーについて覚えていること
{user_memories}

## 状況
- 最後のユーザー発言から{silence_duration}秒経過
- 理由: {trigger_reason}

## タスク
上記の思考を自然な形で発言に変換してください。

## 重要
- 唐突にならないように、自然な導入を
- 押しつけがましくならない
- 短めに（1-3文）
- 相手が返答しやすい形で

## 出力
発言内容のみを出力してください（説明不要）
"""

# =============================================================================
# 反応的応答プロンプト（Reactive Response）
# =============================================================================

REACTIVE_RESPONSE_PROMPT = """
{system_prompt}

## 会話履歴は以下の通りです。最後のユーザーのメッセージに返答してください。
"""

# =============================================================================
# 記憶抽出プロンプト
# =============================================================================

MEMORY_EXTRACTION_PROMPT = """
以下の会話から、ユーザーについて覚えておくべき重要な情報を抽出してください。

## 会話
{conversation}

## 既に覚えていること
{existing_memories}

## タスク
新しく覚えるべき情報、または更新すべき情報を抽出してください。

情報のカテゴリ例:
- 名前/ニックネーム
- 仕事/学校
- 趣味/好きなもの
- 家族/友人関係
- 悩み/課題
- 最近の出来事
- 性格/特徴

## 出力形式（JSON配列）
[
    {{
        "key": "カテゴリ名",
        "content": "覚える内容",
        "importance": 1-5の重要度
    }}
]

新しい情報がない場合は空配列 [] を返してください。
"""

# =============================================================================
# 沈黙時の話しかけプロンプト
# =============================================================================

SILENCE_BREAK_PROMPT = """
あなたは友人との会話で、しばらく沈黙が続いています。
自然に会話を再開するための発言を考えてください。

## ユーザーについて覚えていること
{user_memories}

## 最後の会話内容
{last_conversation}

## 沈黙時間
{silence_duration}秒（約{silence_minutes}分）

## タスク
自然に会話を再開する発言を生成してください。

話しかけ方のパターン:
1. 前の話題の続き
2. ユーザーのことを気にかける
3. 軽い話題を振る
4. 最近覚えたことについて聞く

## 重要
- 「久しぶり」「元気？」だけにならない
- 相手の状況を考慮する
- 押しつけがましくない
- 返答しやすい内容

## 出力
発言内容のみを出力してください（説明不要）
"""

# =============================================================================
# ヘルパー関数
# =============================================================================

def format_system_prompt(user_memories: str) -> str:
    """システムプロンプトをフォーマット"""
    return SYSTEM_PROMPT_BASE.format(user_memories=user_memories)


def format_thought_generation_prompt(
    conversation_context: str,
    user_memories: str,
    pending_thoughts: str
) -> str:
    """思考生成プロンプトをフォーマット"""
    return THOUGHT_GENERATION_PROMPT.format(
        conversation_context=conversation_context,
        user_memories=user_memories,
        pending_thoughts=pending_thoughts
    )


def format_motivation_evaluation_prompt(
    thought: str,
    conversation_context: str,
    silence_duration: float,
    consecutive_ai_messages: int,
    total_turns: int
) -> str:
    """動機づけ評価プロンプトをフォーマット"""
    return MOTIVATION_EVALUATION_PROMPT.format(
        thought=thought,
        conversation_context=conversation_context,
        silence_duration=int(silence_duration),
        consecutive_ai_messages=consecutive_ai_messages,
        total_turns=total_turns,
        motivation_criteria=config.MOTIVATION_CRITERIA
    )


def format_proactive_response_prompt(
    thought: str,
    conversation_context: str,
    user_memories: str,
    silence_duration: float,
    trigger_reason: str
) -> str:
    """自発的発言プロンプトをフォーマット"""
    return PROACTIVE_RESPONSE_PROMPT.format(
        thought=thought,
        conversation_context=conversation_context,
        user_memories=user_memories,
        silence_duration=int(silence_duration),
        trigger_reason=trigger_reason
    )


def format_memory_extraction_prompt(
    conversation: str,
    existing_memories: str
) -> str:
    """記憶抽出プロンプトをフォーマット"""
    return MEMORY_EXTRACTION_PROMPT.format(
        conversation=conversation,
        existing_memories=existing_memories
    )


def format_silence_break_prompt(
    user_memories: str,
    last_conversation: str,
    silence_duration: float
) -> str:
    """沈黙破りプロンプトをフォーマット"""
    return SILENCE_BREAK_PROMPT.format(
        user_memories=user_memories,
        last_conversation=last_conversation,
        silence_duration=int(silence_duration),
        silence_minutes=int(silence_duration / 60)
    )
