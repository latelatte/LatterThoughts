"""
Proactive AI Friend - Discord Bot
Inner Thoughtsフレームワークを実装したDiscord Bot
"""

import asyncio
import uuid
from datetime import datetime

import discord
from discord.ext import commands, tasks

import config
from memory import MemoryManager
from inner_thoughts import InnerThoughtsEngine
from research_logger import ResearchLogger
from information_gatherer import InformationGatherer
from response_classifier import ResponseClassifier


class ProactiveAIBot(commands.Bot):
    """
    Proactive AI Friend Discord Bot
    
    機能:
    - ユーザーメッセージへの反応的応答
    - Inner Thoughtsによる自発的発言
    - 長期記憶によるユーザー理解
    - 研究用ログ収集
    """
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None
        )
        
        # エンジン
        self.engine = InnerThoughtsEngine()
        self.info_gatherer = InformationGatherer()
        self.classifier = ResponseClassifier()
        
        # ユーザーごとの記憶管理
        self.memories: dict[str, MemoryManager] = {}
        
        # セッション管理
        self.session_id = str(uuid.uuid4())[:8]
        self.logger = ResearchLogger(self.session_id)
        
        # アクティブなチャンネル（DMまたは指定チャンネル）
        self.active_channels: set[int] = set()
        
        # 処理中フラグ（二重応答防止）
        self.processing: set[str] = set()
        
        # 情報収集の最終検索時刻
        self.last_info_search: dict[str, datetime] = {}
    
    # =========================================================================
    # イベントハンドラ
    # =========================================================================
    
    async def on_ready(self):
        """Bot起動時"""
        print(f"🤖 {config.AI_NAME} がオンラインになりました！")
        print(f"📊 セッションID: {self.session_id}")
        print(f"⚙️  実験条件: {config.EXPERIMENT_CONDITION}")
        print(f"📝 ログ保存先: {config.LOG_DIRECTORY}/")
        print("-" * 50)
        
        # Proactiveサイクルを開始（既に動いていなければ）
        if config.EXPERIMENT_CONDITION == "proactive":
            if not self.proactive_cycle.is_running():
                self.proactive_cycle.start()
    
    async def on_message(self, message: discord.Message):
        """メッセージ受信時"""
        # Bot自身のメッセージは無視
        if message.author.bot:
            return
        
        # DMまたはメンションされた場合のみ反応
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user in message.mentions
        
        if message.content.startswith(self.command_prefix):
            if is_dm or is_mentioned or message.content.startswith(self.command_prefix):
                await self.process_commands(message)
            return
        
        if not (is_dm or is_mentioned):
            return
        
        # メンションを除去
        content = message.content.replace(f"<@{self.user.id}>", "").strip()
        if not content:
            return
        
        # ユーザーIDとチャンネル
        user_id = str(message.author.id)
        channel_id = message.channel.id
        
        # 二重処理防止
        process_key = f"{user_id}:{message.id}"
        if process_key in self.processing:
            return
        self.processing.add(process_key)
        
        try:
            # 記憶マネージャー取得または作成
            memory = self._get_memory(user_id)
            
            # チャンネルをアクティブに
            self.active_channels.add(channel_id)
            
            # ユーザーメッセージを記録
            memory.add_message("user", content)
            self.logger.log_user_message(user_id, content)
            
            # 応答タイプを判定（リアクション or 返信 or 無視）
            decision = await self.classifier.classify(
                content,
                memory.get_context_summary()
            )
            
            if decision.action == "react" and decision.reaction:
                # リアクションで十分な場合 → 絵文字だけ付ける
                await message.add_reaction(decision.reaction)
                self.logger.log_ai_response(
                    user_id, 
                    f"[reaction: {decision.reaction}]", 
                    is_proactive=False,
                    metadata={"type": "reaction", "reason": decision.reason}
                )
            
            elif decision.action == "reply":
                # 返信が必要な場合 → 通常の応答生成
                async with message.channel.typing():
                    response = await self.engine.generate_reactive_response(memory)
                
                # フォールバックメッセージは履歴に保存しない（文脈が壊れるため）
                is_fallback = "調子悪いみたい..." in response
                if not is_fallback:
                    memory.add_message("assistant", response)
                self.logger.log_ai_response(user_id, response, is_proactive=False)
                
                await message.channel.send(response)
            
            # else: decision.action == "none" なら何もしない
            
            # 定期的に記憶を抽出（5ターンごと）
            if len(memory.short_term) >= 5 and len(memory.short_term) % 5 <= 1:
                await self._extract_and_save_memories(memory)
                
        finally:
            self.processing.discard(process_key)
    
    # =========================================================================
    # Proactiveサイクル
    # =========================================================================
    
    @tasks.loop(seconds=30)
    async def proactive_cycle(self):
        """
        定期的に実行されるProactiveサイクル
        各アクティブユーザーに対して思考生成・評価を行う
        """
        for user_id, memory in list(self.memories.items()):
            try:
                # Proactiveサイクル実行
                response = await self.engine.process_proactive_cycle(memory)
                
                if response:
                    # 発言先のチャンネルを探す
                    channel = await self._get_channel_for_user(user_id)
                    if channel:
                        # 記録
                        memory.add_message("assistant", response)
                        self.logger.log_ai_response(
                            user_id, response, is_proactive=True,
                            metadata={"trigger": "proactive_cycle"}
                        )
                        
                        # 送信
                        await channel.send(response)
                        
            except Exception as e:
                print(f"Proactive cycle error for {user_id}: {e}")
    
    @proactive_cycle.before_loop
    async def before_proactive_cycle(self):
        """Proactiveサイクル開始前にBotの準備完了を待つ"""
        await self.wait_until_ready()
        
        
        # =========================================================================
    # 情報収集サイクル
    # =========================================================================
    
    @tasks.loop(minutes=30)
    async def information_gathering_cycle(self):
        """
        定期的に実行される情報収集サイクル
        各ユーザーの興味に基づいて情報を探し、共有する
        """
        if not config.BRAVE_SEARCH_API_KEY:
            return
        
        now = datetime.now()
        
        for user_id, memory in list(self.memories.items()):
            try:
                # 検索間隔チェック
                last_search = self.last_info_search.get(user_id)
                if last_search:
                    elapsed = (now - last_search).total_seconds()
                    if elapsed < config.SEARCH_INTERVAL:
                        continue
                
                # 長期記憶がないユーザーはスキップ
                if not memory.long_term:
                    continue
                
                # 介入可能かチェック
                if not memory.can_intervene():
                    continue
                
                # 情報を探す
                result = await self.info_gatherer.find_shareable_article(memory)
                
                if result:
                    article, message = result
                    
                    # チャンネルを取得
                    channel = await self._get_channel_for_user(user_id)
                    if channel:
                        # 記録
                        memory.add_message("assistant", message)
                        self.logger.log_ai_response(
                            user_id, message, is_proactive=True,
                            metadata={
                                "trigger": "information_share",
                                "article_title": article.title,
                                "article_url": article.url,
                                "relevance_score": article.relevance_score
                            }
                        )
                        
                        # 送信
                        await channel.send(message)
                
                # 検索時刻を更新
                self.last_info_search[user_id] = now
                        
            except Exception as e:
                print(f"Information gathering error for {user_id}: {e}")
    
    @information_gathering_cycle.before_loop
    async def before_information_gathering_cycle(self):
        """情報収集サイクル開始前にBotの準備完了を待つ"""
        await self.wait_until_ready()
    
    # =========================================================================
    # コマンド
    # =========================================================================
    
    async def setup_hook(self):
        """コマンドの設定"""
        
        @self.command(name="status")
        async def status(ctx):
            """現在のステータスを表示"""
            user_id = str(ctx.author.id)
            memory = self._get_memory(user_id)
            
            stats = self.logger.get_thought_statistics()
            metrics = self.logger.calculate_metrics(user_id)
            
            embed = discord.Embed(
                title=f"📊 {config.AI_NAME} ステータス",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="セッション",
                value=f"ID: `{self.session_id}`\n条件: `{config.EXPERIMENT_CONDITION}`",
                inline=False
            )
            
            embed.add_field(
                name="会話統計",
                value=f"総ターン: {metrics.total_turns}\n"
                      f"あなたのメッセージ: {metrics.user_messages}\n"
                      f"私の返答: {metrics.ai_reactive_responses}\n"
                      f"私からの話しかけ: {metrics.ai_proactive_interventions}",
                inline=True
            )
            
            if stats:
                embed.add_field(
                    name="思考統計",
                    value=f"生成された思考: {stats.get('total_thoughts', 0)}\n"
                          f"発言された思考: {stats.get('expressed_thoughts', 0)}\n"
                          f"平均動機スコア: {stats.get('avg_motivation_score', 0):.2f}",
                    inline=True
                )
            
            embed.add_field(
                name="記憶",
                value=f"短期記憶: {len(memory.short_term)}ターン\n"
                      f"長期記憶: {len(memory.long_term)}項目\n"
                      f"保留中の思考: {len(memory.thought_reservoir)}個",
                inline=True
            )
            
            await ctx.send(embed=embed)
        
        @self.command(name="memories")
        async def show_memories(ctx):
            """覚えていることを表示"""
            user_id = str(ctx.author.id)
            memory = self._get_memory(user_id)
            
            memories_text = memory.get_all_memories_summary()
            
            embed = discord.Embed(
                title=f"🧠 {ctx.author.display_name}さんについて覚えていること",
                description=memories_text if memories_text else "まだ何も覚えていません",
                color=discord.Color.green()
            )
            
            await ctx.send(embed=embed)
        
        @self.command(name="forget")
        async def forget_memories(ctx):
            """記憶をリセット"""
            user_id = str(ctx.author.id)
            if user_id in self.memories:
                del self.memories[user_id]
            
            await ctx.send("記憶をリセットしました。また一から仲良くなりましょう！")
        
        @self.command(name="thoughts")
        async def show_thoughts(ctx):
            """保留中の思考を表示（デバッグ用）"""
            user_id = str(ctx.author.id)
            memory = self._get_memory(user_id)
            
            pending = memory.get_pending_thoughts()
            
            if not pending:
                await ctx.send("今のところ、特に言いたいことはないかな")
                return
            
            embed = discord.Embed(
                title="💭 今考えていること",
                color=discord.Color.purple()
            )
            
            threshold = config.MOTIVATION_THRESHOLD
            for i, thought in enumerate(pending[:5], 1):
                # 閾値を超えているかどうかでマークを付ける
                passed = "✅" if thought.motivation_score >= threshold else "❌"
                
                # 思考内容と理由を表示
                value = f"{thought.content[:100]}"
                if thought.reasoning:
                    value += f"\n📊 **理由**: {thought.reasoning[:100]}"
                value += f"\n{passed} 閾値({threshold}) {'超え' if thought.motivation_score >= threshold else '未満'}"
                
                embed.add_field(
                    name=f"思考 {i} (スコア: {thought.motivation_score:.1f})",
                    value=value,
                    inline=False
                )
            
            await ctx.send(embed=embed)
        
        @self.command(name="config")
        async def show_config(ctx):
            """設定を表示"""
            embed = discord.Embed(
                title="⚙️ 現在の設定",
                color=discord.Color.orange()
            )
            
            embed.add_field(
                name="Proactive設定",
                value=f"動機づけ閾値: {config.MOTIVATION_THRESHOLD}\n"
                      f"思考生成間隔: {config.THOUGHT_GENERATION_INTERVAL}秒\n"
                      f"沈黙タイムアウト: {config.SILENCE_TIMEOUT}秒\n"
                      f"最大連続発言: {config.MAX_CONSECUTIVE_INTERVENTIONS}回",
                inline=False
            )
            
            embed.add_field(
                name="情報収集設定",
                value=f"有効: {config.ENABLE_INFORMATION_GATHERING}\n"
                      f"検索間隔: {config.SEARCH_INTERVAL}秒\n"
                      f"共有閾値: {config.INFO_SHARE_MOTIVATION_THRESHOLD}\n"
                      f"1日の最大共有: {config.MAX_DAILY_SHARES}回",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        @self.command(name="interests")
        async def show_interests(ctx):
            """推測された興味を表示"""
            user_id = str(ctx.author.id)
            memory = self._get_memory(user_id)
            
            async with ctx.typing():
                interests = await self.info_gatherer.extract_interests(memory)
            
            if interests:
                embed = discord.Embed(
                    title="🔍 あなたの興味（推測）",
                    description="\n".join([f"• {i}" for i in interests]),
                    color=discord.Color.blue()
                )
                embed.set_footer(text="これらのキーワードで情報を探しています")
            else:
                embed = discord.Embed(
                    title="🔍 あなたの興味",
                    description="まだ興味を把握できていません。もっとお話ししましょう！",
                    color=discord.Color.light_grey()
                )
            
            await ctx.send(embed=embed)
        
        @self.command(name="search")
        async def search_now(ctx):
            """今すぐ情報を探して共有"""
            if not config.BRAVE_SEARCH_API_KEY:
                await ctx.send("🔍 検索機能が設定されていません（BRAVE_SEARCH_API_KEY が必要です）")
                return
            
            user_id = str(ctx.author.id)
            memory = self._get_memory(user_id)
            
            await ctx.send("🔍 あなたが興味ありそうな情報を探しています...")
            
            async with ctx.typing():
                result = await self.info_gatherer.find_shareable_article(memory)
            
            if result:
                article, message = result
                
                # 記録
                memory.add_message("assistant", message)
                self.logger.log_ai_response(
                    user_id, message, is_proactive=True,
                    metadata={
                        "trigger": "manual_search",
                        "article_title": article.title,
                        "relevance_score": article.relevance_score
                    }
                )
                
                await ctx.send(message)
            else:
                await ctx.send("今回は特に良さそうな情報が見つからなかったかも...また後で探してみるね！")
                
        @self.command(name="export")
        async def export_logs(ctx):
            """ログをエクスポート"""
            self.logger.save_session_summary()
            
            summary = self.logger.export_session_summary()
            
            embed = discord.Embed(
                title="📤 ログをエクスポートしました",
                color=discord.Color.gold()
            )
            
            embed.add_field(
                name="保存先",
                value=f"`{config.LOG_DIRECTORY}/`",
                inline=False
            )
            
            embed.add_field(
                name="サマリー",
                value=f"```json\n{str(summary)[:500]}...\n```",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        @self.command(name="clear")
        async def clear_messages(ctx, limit: int = 50):
            """BotのDMメッセージを削除"""
            if not isinstance(ctx.channel, discord.DMChannel):
                await ctx.send("DMでのみ使えます")
                return
            
            deleted = 0
            async for message in ctx.channel.history(limit=limit):
                if message.author == self.user:
                    await message.delete()
                    deleted += 1
                    await asyncio.sleep(0.5)  # レート制限対策
            
            await ctx.send(f"🧹 {deleted}件削除しました", delete_after=5)
    
    # =========================================================================
    # ヘルパー関数
    # =========================================================================
    
    def _get_memory(self, user_id: str) -> MemoryManager:
        """ユーザーの記憶マネージャーを取得または作成"""
        if user_id not in self.memories:
            self.memories[user_id] = MemoryManager(user_id)
        return self.memories[user_id]
    
    async def _get_channel_for_user(self, user_id: str):
        """ユーザーに送信するチャンネルを取得"""
        try:
            user = await self.fetch_user(int(user_id))
            return user.dm_channel or await user.create_dm()
        except Exception:
            return None
    
    async def _extract_and_save_memories(self, memory: MemoryManager):
        """記憶を抽出して保存"""
        try:
            print(f"[DEBUG] Extracting memories...")
            new_memories = await self.engine.extract_memories(memory)
            print(f"[DEBUG] Extracted {len(new_memories)} memories: {new_memories}")
            for mem in new_memories:
                memory.add_long_term_memory(
                    key=mem.get("key", "その他"),
                    content=mem.get("content", ""),
                    importance=mem.get("importance", 3.0)
                )
        except Exception as e:
            print(f"Memory extraction error: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# メイン
# =============================================================================

def main():
    """Botを起動"""
    if not config.DISCORD_TOKEN:
        print("❌ DISCORD_TOKEN が設定されていません")
        print("   .env ファイルに DISCORD_TOKEN=your_token を追加してください")
        return
    
    if not config.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY が設定されていません")
        print("   .env ファイルに OPENAI_API_KEY=your_key を追加してください")
        return
    
    bot = ProactiveAIBot()
    bot.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()
