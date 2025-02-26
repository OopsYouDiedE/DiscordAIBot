import discord
from discord.ext import commands, tasks
import random
import json
import os
import asyncio
import datetime
import re
import logging
import nltk
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
from openai import AsyncOpenAI
import sys
import traceback
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 下载必要的NLTK数据
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('punkt_tab')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('discord_bot')

# 机器人配置
TOKEN = os.getenv('DISCORD_TOKEN')
PREFIX = '!'
MEMORY_FILE = 'memory.json'
CONVERSATION_HISTORY_FILE = 'conversation_history.json'

# LLM配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 替换为你的API密钥
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v3")

# Google搜索配置
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# 初始化机器人
intents = discord.Intents.all()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# 情感分析器
sia = SentimentIntensityAnalyzer()

# 初始化OpenAI客户端
a_client = AsyncOpenAI(api_key=OPENAI_API_KEY,
                       base_url=OPENAI_BASE_URL).chat.completions


# 机器人状态和记忆
class BotMemory:

    def __init__(self):
        self.user_data = {}
        self.conversation_history = defaultdict(list)
        self.group_interests = Counter()
        self.active_topics = {}
        self.bot_mood = "neutral"
        self.last_interaction = {}
        self.load_memory()
        self.load_conversation_history()

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_data = data.get('user_data', {})
                    self.group_interests = Counter(
                        data.get('group_interests', {}))
                    self.active_topics = data.get('active_topics', {})
                    self.bot_mood = data.get('bot_mood', "neutral")
                    self.last_interaction = data.get('last_interaction', {})
                logger.info("记忆数据已加载")
            except Exception as e:
                logger.error(f"加载记忆数据时出错: {e}")
                self.user_data = {}
                self.group_interests = Counter()
                self.active_topics = {}
                self.bot_mood = "neutral"
                self.last_interaction = {}

    def save_memory(self):
        try:
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'user_data': self.user_data,
                        'group_interests': dict(self.group_interests),
                        'active_topics': self.active_topics,
                        'bot_mood': self.bot_mood,
                        'last_interaction': self.last_interaction
                    },
                    f,
                    ensure_ascii=False,
                    indent=2)
            logger.info("记忆数据已保存")
        except Exception as e:
            logger.error(f"保存记忆数据时出错: {e}")

    def load_conversation_history(self):
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            try:
                with open(CONVERSATION_HISTORY_FILE, 'r',
                          encoding='utf-8') as f:
                    self.conversation_history = defaultdict(list, json.load(f))
                logger.info("对话历史已加载")
            except Exception as e:
                logger.error(f"加载对话历史时出错: {e}")
                self.conversation_history = defaultdict(list)

    def save_conversation_history(self):
        try:
            with open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(dict(self.conversation_history),
                          f,
                          ensure_ascii=False,
                          indent=2)
            logger.info("对话历史已保存")
        except Exception as e:
            logger.error(f"保存对话历史时出错: {e}")

    def add_user_interaction(self, user_id, username, message_content,
                             channel_id):
        # 确保用户在数据库中
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                'username': username,
                'first_seen': datetime.datetime.now().isoformat(),
                'interaction_count': 0,
                'topics': [],
                'sentiment': "neutral",
                'last_message': "",
            }

        # 更新用户数据
        self.user_data[user_id]['interaction_count'] += 1
        self.user_data[user_id]['last_message'] = message_content
        self.user_data[user_id]['last_interaction'] = datetime.datetime.now(
        ).isoformat()

        # 分析消息情感
        sentiment = sia.polarity_scores(message_content)
        if sentiment['compound'] > 0.3:
            self.user_data[user_id]['sentiment'] = "positive"
        elif sentiment['compound'] < -0.3:
            self.user_data[user_id]['sentiment'] = "negative"
        else:
            self.user_data[user_id]['sentiment'] = "neutral"

        # 提取可能的话题
        words = nltk.word_tokenize(message_content.lower())
        nouns = [word for word in words if len(word) > 3]  # 简单假设长词可能是话题

        # 更新用户话题
        if nouns:
            if 'topics' not in self.user_data[user_id]:
                self.user_data[user_id]['topics'] = []
            self.user_data[user_id]['topics'].extend(nouns)
            self.user_data[user_id]['topics'] = self.user_data[user_id][
                'topics'][-20:]  # 保留最近20个话题

            # 更新群组兴趣
            self.group_interests.update(nouns)

        # 更新对话历史
        channel_key = str(channel_id)
        if channel_key not in self.conversation_history:
            self.conversation_history[channel_key] = []

        self.conversation_history[channel_key].append({
            'user_id':
            user_id,
            'username':
            username,
            'content':
            message_content,
            'timestamp':
            datetime.datetime.now().isoformat()
        })

        # 保持对话历史在合理大小
        if len(self.conversation_history[channel_key]) > 100:
            self.conversation_history[channel_key] = self.conversation_history[
                channel_key][-100:]

        # 更新最后交互时间
        self.last_interaction[user_id] = datetime.datetime.now().isoformat()

        # 保存更新
        self.save_memory()
        self.save_conversation_history()

    def get_recent_topics(self, limit=5):
        """获取最近的热门话题"""
        return [
            topic for topic, count in self.group_interests.most_common(limit)
        ]

    def get_user_info(self, user_id):
        """获取用户信息"""
        return self.user_data.get(user_id, {})

    def get_channel_context(self, channel_id, limit=10):
        """获取频道最近的对话上下文"""
        channel_key = str(channel_id)
        if channel_key in self.conversation_history:
            return self.conversation_history[channel_key][-limit:]
        return []


# 初始化机器人记忆
memory = BotMemory()


# LLM集成
async def ask_llm(query, context=None, system_prompt=None):
    """使用LLM生成回复"""
    messages = []

    # 添加系统提示
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    # 添加上下文
    if context:
        for message in context:
            messages.append({
                'role':
                'user' if message.get('user_id') != 'bot' else 'assistant',
                'content':
                f"{message.get('username', 'User')}: {message.get('content', '')}"
            })

    # 添加当前问题
    messages.append({'role': 'user', 'content': query})

    try:
        # 创建LLM请求
        completion = await a_client.create(model=LLM_MODEL, messages=messages)
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM请求错误: {e}")
        return None


# Google搜索集成
def google_search(query, api_key=GOOGLE_API_KEY, cx=GOOGLE_CX, num=5):
    """使用Google自定义搜索API进行搜索"""
    try:
        if not api_key or not cx:
            logger.error("Google API设置不完整")
            return None

        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": api_key, "cx": cx, "q": query, "num": num}

        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        if "items" in results:
            return results["items"]
        return None
    except Exception as e:
        logger.error(f"Google搜索错误: {e}")
        return None


def display_search_results(results, max_results=3):
    """格式化搜索结果"""
    if not results:
        return "抱歉，我找不到相关信息。"

    formatted_results = "以下是我找到的信息：\n\n"

    for i, item in enumerate(results[:max_results], 1):
        title = item.get('title', 'No Title')
        snippet = item.get('snippet', 'No Description').replace('\n', ' ')
        link = item.get('link', '#')

        # 简化URL显示
        domain = re.search(r'//([^/]+)', link)
        domain = domain.group(1) if domain else link

        formatted_results += f"{i}. **{title}**\n{snippet}\n来源: {domain}\n\n"

    return formatted_results


# 机器人响应生成
class ResponseGenerator:

    def __init__(self, memory):
        self.memory = memory
        self.greetings = [
            "你好啊！", "嗨！", "大家好！", "有人在吗？", "今天过得怎么样？", "打扰一下~", "我回来了！",
            "有什么有趣的事情发生吗？"
        ]
        self.reactions = ["👋", "❤️", "👍", "😊", "🎉", "🤔", "😂", "🙌", "✨", "🔥"]
        self.topic_starters = [
            "我最近在想，{topic}是不是很有意思？", "大家对{topic}有什么看法？",
            "说到{topic}，我有个问题想问大家...", "我发现{topic}真的很吸引人，你们觉得呢？",
            "有人了解{topic}吗？我想了解更多。"
        ]
        self.questions = [
            "你们觉得呢？", "有人有不同意见吗？", "大家有什么想法？", "这个话题你们感兴趣吗？", "有人能分享一下经验吗？"
        ]

        # 知识库 - 可以扩展
        self.knowledge_base = {
            "discord": "Discord是一个专为社区设计的免费语音、视频和文字聊天应用程序。",
            "python": "Python是一种解释型、高级、通用型编程语言，由吉多·范罗苏姆创造于1989年。",
            "游戏": "游戏是一种通过电子设备进行的娱乐活动，可以是单人或多人参与的。",
            "电影": "电影是一种视觉艺术形式，通过连续的图像创造幻觉，讲述故事或表达思想。",
            "音乐": "音乐是一种艺术形式，通过有组织的声音和静默来创造美的形式。",
            "编程": "编程是编写计算机程序的过程，这些程序是计算机执行特定任务的指令集。",
            "人工智能": "人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能的系统。",
            "机器学习": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。"
        }

    def generate_greeting(self):
        """生成问候语"""
        return random.choice(self.greetings)

    def generate_reaction(self):
        """生成表情反应"""
        return random.choice(self.reactions)

    def generate_topic(self):
        """生成新话题"""
        topics = self.memory.get_recent_topics()
        if not topics:
            topics = list(self.knowledge_base.keys())

        topic = random.choice(topics)
        return random.choice(self.topic_starters).format(topic=topic)

    def generate_question(self):
        """生成问题"""
        return random.choice(self.questions)

    async def answer_question(self, question, channel_id):
        """回答问题 - 使用LLM和搜索引擎"""
        # 先检查本地知识库
        for keyword, answer in self.knowledge_base.items():
            if keyword.lower() in question.lower():
                # 知识库中有答案，用LLM扩展一下
                try:
                    enhanced_answer = await ask_llm(
                        f"基于以下信息回答问题。信息: {answer}，问题: {question}")
                    if enhanced_answer:
                        return enhanced_answer
                    return answer
                except:
                    return answer

        # 尝试使用搜索引擎查找答案
        search_results = None
        llm_answer = None
        final_answer = None

        # 判断问题是否需要搜索最新信息
        needs_search = any(
            word in question.lower()
            for word in ['最新', '最近', '新闻', '现在', '今天', '昨天', '本周', '本月', '当前'])

        if needs_search:
            # 获取搜索结果
            search_results = google_search(question)

            if search_results:
                formatted_results = display_search_results(search_results)

                # 将搜索结果提供给LLM进行总结
                context_for_llm = self.memory.get_channel_context(channel_id,
                                                                  limit=5)

                try:
                    llm_answer = await ask_llm(
                        f"根据以下搜索结果和上下文信息，回答用户问题。问题: {question}\n\n搜索结果:\n{formatted_results}",
                        context=context_for_llm,
                        system_prompt=
                        "你是一个友好的Discord群友，正在参与群聊。你需要根据提供的搜索结果回答问题，回答要简洁自然，像普通群友一样说话。"
                    )
                    final_answer = llm_answer
                except Exception as e:
                    logger.error(f"使用LLM处理搜索结果时出错: {e}")
                    final_answer = f"这是我找到的一些资料：\n\n{formatted_results}"

        # 如果不需要搜索或搜索失败，直接使用LLM回答
        if not final_answer:
            context_for_llm = self.memory.get_channel_context(channel_id,
                                                              limit=5)
            try:
                llm_answer = await ask_llm(
                    question,
                    context=context_for_llm,
                    system_prompt=
                    "你是一个友好的Discord群友，正在参与群聊。回答要简洁自然，像普通群友一样说话。不要使用太正式或机器人式的语言。如果不确定答案，就坦率地说不知道，可以适当加入表情符号增加亲和力。"
                )
                final_answer = llm_answer
            except Exception as e:
                logger.error(f"使用LLM回答问题时出错: {e}")
                final_answer = "这是个好问题！我不太确定答案，但我们可以一起讨论一下。"

        # 如果所有方法都失败，返回默认回复
        if not final_answer:
            return "这是个好问题！我不太确定答案，但我们可以一起讨论一下。"

        return final_answer

    async def generate_comment(self, message_content, context=None):
        """根据消息内容生成评论 - 使用LLM增强"""
        sentiment = sia.polarity_scores(message_content)

        # 尝试使用LLM生成更自然的评论
        try:
            comment = await ask_llm(
                f"对以下消息提供一个简短、自然的回复，像普通朋友一样说话：\n{message_content}",
                context=context,
                system_prompt=
                "你是一个友好的Discord群友。你的回复应该简短（不超过30个字），自然，像普通朋友一样说话。不要显得太正式或机器人式。"
            )
            if comment:
                return comment
        except Exception as e:
            logger.error(f"使用LLM生成评论时出错: {e}")

        # 如果LLM失败，使用基于情感的模板回复
        if sentiment['compound'] > 0.5:
            return random.choice(
                ["我完全同意你的观点！", "说得太好了！", "这个想法真棒！", "我也是这么想的！", "你的观点很有见地！"])
        elif sentiment['compound'] < -0.5:
            return random.choice([
                "听起来有点困难啊...", "希望情况能变得更好。", "这确实是个问题，有什么我能帮忙的吗？",
                "我理解你的感受，要不要聊聊别的？", "也许事情会好转的。"
            ])
        else:
            return random.choice(
                ["有意思的观点。", "我明白你的意思了。", "这让我想到了...", "谢谢分享！", "继续说下去？"])

    async def personalize_response(self, user_id, base_response):
        """根据用户信息个性化响应"""
        user_info = self.memory.get_user_info(user_id)

        if not user_info:
            return base_response

        # 如果是熟悉的用户，可能会提到他们的兴趣
        if user_info.get('interaction_count', 0) > 10:
            user_topics = user_info.get('topics', [])
            if user_topics:
                recent_topic = random.choice(user_topics)
                # 尝试使用LLM生成更自然的个性化回复
                try:
                    personalized = await ask_llm(
                        f"请基于以下基础回复和用户兴趣创建一个个性化回复。基础回复：{base_response}，用户兴趣：{recent_topic}",
                        system_prompt=
                        "你是一个友好的Discord群友，正在与熟悉的朋友聊天。请保持回复简短自然，类似于普通用户的聊天方式，不要显得太正式。可以适当提及用户的兴趣爱好。"
                    )
                    if personalized:
                        return personalized
                except:
                    pass

                # 如果LLM失败，使用模板
                personalized_responses = [
                    f"{base_response} 对了，你不是对{recent_topic}很感兴趣吗？",
                    f"{base_response} 话说回来，最近{recent_topic}有什么新进展吗？",
                    f"作为一个喜欢{recent_topic}的人，你觉得{base_response}"
                ]
                return random.choice(personalized_responses)

        return base_response

    async def generate_followup(self, context, channel_id=None):
        """根据上下文生成后续回复"""
        if not context:
            return self.generate_topic()

        # 分析最近的对话
        last_message = context[-1]['content']

        # 检测是否是问题
        if '?' in last_message or '？' in last_message:
            return await self.answer_question(last_message, channel_id)

        # 使用LLM生成更自然的跟进
        try:
            followup = await ask_llm(
                "请根据上述对话生成一个自然的跟进回复",
                context=context,
                system_prompt=
                "你是Discord群组中的一个普通成员。基于上下文提供简短、自然的跟进，像普通群友一样说话。不要使用太正式或机器人式的语言。"
            )
            if followup:
                return followup
        except Exception as e:
            logger.error(f"使用LLM生成跟进回复时出错: {e}")

        # 如果LLM失败，使用简单评论
        return await self.generate_comment(last_message)


# 初始化响应生成器
response_generator = ResponseGenerator(memory)


# 机器人事件处理
@bot.event
async def on_ready():
    logger.info(f'{bot.user.name} 已连接到Discord!')
    change_activity.start()
    periodic_interaction.start()
    save_data.start()
    await bot.change_presence(activity=discord.Game(name="初次见面，请多指教！"))

    # 向所有可见频道发送问候
    for guild in bot.guilds:
        # 寻找合适的文本频道
        general_channels = [
            ch for ch in guild.text_channels if "general" in ch.name.lower()
        ]
        if general_channels:
            channel = general_channels[0]
        else:
            # 如果没有general频道，选择第一个文本频道
            text_channels = [
                ch for ch in guild.text_channels
                if ch.permissions_for(guild.me).send_messages
            ]
            if text_channels:
                channel = text_channels[0]
            else:
                continue

        try:
            await channel.send(
                "大家好！我是新加入的虚拟群友，可以和我聊天，问我问题，或者用`!help`查看我的功能。期待和大家成为好朋友！😊")
        except Exception as e:
            logger.error(f"发送问候消息时出错: {e}")


@bot.event
async def on_message(message):
    # 忽略自己的消息
    if message.author == bot.user:
        return

    # 记录用户交互
    memory.add_user_interaction(str(message.author.id), message.author.name,
                                message.content, str(message.channel.id))

    # 如果消息以命令前缀开头，处理命令
    if message.content.startswith(PREFIX):
        await bot.process_commands(message)
        return

    # 决定是否回复
    should_reply = False

    # 如果被提及，100%回复
    if bot.user.mentioned_in(message):
        should_reply = True
    # 如果是问题，70%概率回复
    elif '?' in message.content or '？' in message.content:
        should_reply = random.random() < 0.7
    # 如果是常规消息，30%概率回复
    else:
        should_reply = random.random() < 0.3

    # 如果决定回复，处理消息
    if should_reply:
        async with message.channel.typing():  # 显示"正在输入"状态
            await process_message(message)

    # 独立于回复决策的表情反应，20%概率
    if random.random() < 0.2:
        try:
            await message.add_reaction(response_generator.generate_reaction())
        except:
            pass


async def process_message(message):
    """处理消息并生成回复"""
    try:
        # 获取频道上下文
        context = memory.get_channel_context(str(message.channel.id))

        # 准备回复
        reply = None
        typing_delay = 1  # 默认输入延迟

        # 如果被提及，直接回复
        if bot.user.mentioned_in(message):
            # 处理消息中提到机器人的情况
            content = re.sub(f'<@!?{bot.user.id}>', '',
                             message.content).strip()
            if not content:  # 如果只是提到了机器人，没有实际内容
                content = "你好"

            # 使用LLM生成回复
            context_for_llm = get_context_for_llm(context)
            typing_delay = min(2 + len(content) * 0.01, 4)  # 根据内容长度调整"输入"时间

            try:
                reply = await ask_llm(
                    content,
                    context=context_for_llm,
                    system_prompt=
                    "你是Discord群组中的一个友好成员。你应该提供简短、自然的回复，就像普通群友一样说话。不要使用太正式或机器人式的语言。如果被问到问题，尽量提供有帮助的回答，但保持对话风格轻松自然。"
                )
            except Exception as e:
                logger.error(f"使用LLM生成回复出错: {e}")
                reply = await response_generator.generate_followup(
                    context, message.channel.id)

        # 如果是问题，使用问题处理逻辑
        elif '?' in message.content or '？' in message.content:
            typing_delay = min(2 + len(message.content) * 0.01,
                               5)  # 问题可能需要更长的"思考"时间
            reply = await response_generator.answer_question(
                message.content, message.channel.id)

        # 长消息，生成评论
        elif len(message.content) > 50:
            typing_delay = min(1.5 + len(message.content) * 0.005,
                               3)  # 评论不需要太长的思考时间
            context_for_llm = get_context_for_llm(context)
            reply = await response_generator.generate_comment(
                message.content, context_for_llm)

        # 短消息处理
        else:
            typing_delay = 1  # 短消息快速回复
            # 是否是问候
            if any(greeting in message.content.lower()
                   for greeting in ['hello', 'hi', '你好', '嗨']):
                reply = response_generator.generate_greeting()
            else:
                # 50%概率生成评论，50%概率提出新话题
                if random.random() < 0.5:
                    context_for_llm = get_context_for_llm(context)
                    reply = await response_generator.generate_comment(
                        message.content, context_for_llm)
                else:
                    reply = response_generator.generate_topic()

        # 如果所有方法都失败，使用一个安全的默认回复
        if not reply:
            reply = "嗯，有意思。你们继续，我先看看。"

        # 个性化响应（对熟悉的用户）
        if random.random() < 0.7:  # 70%的概率进行个性化
            reply = await response_generator.personalize_response(
                str(message.author.id), reply)

        # 模拟输入时间
        await asyncio.sleep(typing_delay)

        # 发送回复（有50%概率使用reply，50%概率使用普通消息）
        if bot.user.mentioned_in(message) or random.random() < 0.5:
            await message.reply(reply)
        else:
            await message.channel.send(reply)

    except Exception as e:
        logger.error(f"处理消息时出错: {traceback.format_exc()}")
        try:
            await message.channel.send("抱歉，我刚走神了，能再说一遍吗？")
        except:
            pass


def get_context_for_llm(context, limit=5):
    """将上下文转换为LLM可用的格式"""
    if not context:
        return []

    formatted_context = []
    for msg in context[-limit:]:
        formatted_context.append({
            'user_id': msg.get('user_id', ''),
            'username': msg.get('username', 'User'),
            'content': msg.get('content', '')
        })

    return formatted_context


# 定时任务
@tasks.loop(minutes=30)
async def change_activity():
    """定期更改机器人状态"""
    activities = [
        discord.Game(name="思考人生"),
        discord.Activity(type=discord.ActivityType.listening, name="群友讨论"),
        discord.Activity(type=discord.ActivityType.watching, name="有趣的对话"),
        discord.Game(name="学习新知识"),
        discord.Activity(type=discord.ActivityType.competing, name="智力竞赛")
    ]

    activity = random.choice(activities)
    await bot.change_presence(activity=activity)
    logger.info(f"已更改活动状态为: {activity.name}")


@tasks.loop(hours=3)
async def periodic_interaction():
    """定期在活跃频道发起互动"""
    # 获取所有文本频道
    for guild in bot.guilds:
        text_channels = [
            channel for channel in guild.channels
            if isinstance(channel, discord.TextChannel)
            and channel.permissions_for(guild.me).send_messages
        ]

        if not text_channels:
            continue

        # 选择一个随机频道
        channel = random.choice(text_channels)

        # 获取频道上下文
        context = memory.get_channel_context(str(channel.id))

        # 如果该频道24小时内有活跃对话，有更高概率互动
        recent_activity = any(
            datetime.datetime.fromisoformat(msg['timestamp']) >
            datetime.datetime.now() - datetime.timedelta(hours=24)
            for msg in context) if context else False

        if recent_activity and random.random() < 0.7:
            # 生成一个新话题或跟进现有对话
            try:
                async with channel.typing():
                    if random.random() < 0.6:
                        # 提出新话题
                        topic_starter = response_generator.generate_topic()
                        # 使用LLM扩展话题以增加深度
                        enhanced_topic = await ask_llm(
                            f"请基于这个话题启动语'{topic_starter}'创建一个更自然、有深度的话题启动消息，要简洁自然，像普通群友发起的话题一样。"
                        )
                        message = enhanced_topic if enhanced_topic else topic_starter

                        # 添加问题以促进互动
                        if random.random() < 0.7:
                            message += " " + response_generator.generate_question(
                            )
                    else:
                        # 跟进最近的对话
                        message = await response_generator.generate_followup(
                            context, channel.id)

                    # 等待模拟打字时间
                    await asyncio.sleep(min(len(message) * 0.05, 3))
                    await channel.send(message)
                    logger.info(f"在频道 {channel.name} 发起了互动")
            except Exception as e:
                logger.error(f"在频道 {channel.name} 发送消息时出错: {e}")


@tasks.loop(minutes=15)
async def save_data():
    """定期保存数据"""
    memory.save_memory()
    memory.save_conversation_history()


# 命令处理
@bot.command(name='help1', help='显示帮助信息')
async def help_command(ctx):
    embed = discord.Embed(title="虚拟群友机器人帮助",
                          description="我是一个模拟真实群友行为的机器人，以下是我的一些功能：",
                          color=discord.Color.blue())

    embed.add_field(name="自然交流",
                    value="我会自动回复消息、提出话题、参与讨论，就像普通群友一样",
                    inline=False)

    embed.add_field(name="搜索能力", value="如果你问我问题，我会尝试搜索网络找到答案", inline=False)

    embed.add_field(name="智能对话", value="我可以理解上下文，记住群友信息，提供个性化回复", inline=False)

    embed.add_field(name="命令",
                    value=f"""
        `{PREFIX}help` - 显示此帮助信息
        `{PREFIX}topic` - 我会提出一个新话题
        `{PREFIX}mood` - 查看我当前的心情
        `{PREFIX}stats` - 显示群组统计信息
        `{PREFIX}ask [问题]` - 向我咨询任何问题
        `{PREFIX}search [关键词]` - 搜索特定信息
        """,
                    inline=False)

    embed.set_footer(text="@提及我或直接发消息，我都会尝试回复！")

    await ctx.send(embed=embed)


@bot.command(name='topic', help='提出一个新话题')
async def topic_command(ctx):
    async with ctx.typing():
        topic = response_generator.generate_topic()

        # 使用LLM增强话题
        try:
            enhanced_topic = await ask_llm(
                f"基于'{topic}'创建一个更自然有趣的话题启动，像真实群友一样。保持简短自然。")
            if enhanced_topic:
                topic = enhanced_topic
        except Exception as e:
            logger.error(f"增强话题时出错: {e}")

        question = response_generator.generate_question()
        await asyncio.sleep(1)  # 模拟输入延迟
        await ctx.send(f"{topic} {question}")


@bot.command(name='mood', help='查看机器人当前心情')
async def mood_command(ctx):
    moods = {
        "happy": "我现在心情很好！😊",
        "neutral": "我现在心情平静，一切都挺好的。😌",
        "excited": "我现在超级兴奋！有什么好玩的事情吗？🤩",
        "curious": "我对一切都充满好奇心！有什么新鲜事？🧐",
        "tired": "说实话，我有点累了...但还是很乐意聊天！😴",
        "playful": "我现在心情很调皮，想找点乐子！😜"
    }

    # 随机选择一个心情
    current_mood = random.choice(list(moods.keys()))
    memory.bot_mood = current_mood

    # 使用LLM生成更自然的心情描述
    try:
        mood_description = await ask_llm(
            f"你当前的心情是{current_mood}，请像一个普通的Discord群友一样，用一两句话描述你现在的心情状态。要简短自然，加入适合的表情符号。"
        )
        if mood_description:
            await ctx.send(mood_description)
            return
    except:
        pass

    # 默认回复
    await ctx.send(moods[current_mood])


@bot.command(name='stats', help='显示群组统计信息')
async def stats_command(ctx):
    embed = discord.Embed(title="群组统计信息",
                          description="以下是我收集的一些群组数据：",
                          color=discord.Color.green())

    # 热门话题
    hot_topics = memory.get_recent_topics(5)
    if hot_topics:
        embed.add_field(name="热门话题",
                        value="\n".join([f"• {topic}"
                                         for topic in hot_topics]),
                        inline=False)

    # 活跃用户
    active_users = sorted([(user_id, data)
                           for user_id, data in memory.user_data.items()],
                          key=lambda x: x[1].get('interaction_count', 0),
                          reverse=True)[:5]

    if active_users:
        embed.add_field(
            name="活跃用户",
            value="\n".join([
                f"• {data['username']} ({data.get('interaction_count', 0)}条消息)"
                for user_id, data in active_users
            ]),
            inline=False)

    # 总体统计
    total_messages = sum(
        data.get('interaction_count', 0) for data in memory.user_data.values())
    embed.add_field(
        name="总体统计",
        value=f"• 记录用户数: {len(memory.user_data)}\n• 总消息数: {total_messages}",
        inline=False)

    await ctx.send(embed=embed)


@bot.command(name='ask', help='向机器人提问')
async def ask_command(ctx, *, question):
    async with ctx.typing():
        # 增加输入延迟以模拟思考
        typing_delay = min(2 + len(question) * 0.01, 5)
        await asyncio.sleep(typing_delay)

        # 获取频道上下文
        context = memory.get_channel_context(str(ctx.channel.id))

        # 使用响应生成器回答问题
        answer = await response_generator.answer_question(
            question, ctx.channel.id)
        await ctx.reply(answer)


@bot.command(name='search', help='搜索特定信息')
async def search_command(ctx, *, query):
    async with ctx.typing():
        # 增加输入延迟以模拟搜索
        await asyncio.sleep(2)

        # 使用Google搜索
        results = google_search(query)

        if not results:
            await ctx.reply("抱歉，我搜索不到相关信息。你可以尝试换个关键词。")
            return

        # 格式化搜索结果
        formatted_results = display_search_results(results)

        # 使用LLM总结搜索结果
        try:
            summary = await ask_llm(
                f"请根据以下搜索结果，总结对'{query}'的回答。以自然对话方式回复，不要重复'根据搜索结果'之类的话。\n\n{formatted_results}"
            )
            if summary:
                await ctx.reply(summary)
                return
        except Exception as e:
            logger.error(f"总结搜索结果时出错: {e}")

        # 如果LLM总结失败，直接发送格式化结果
        await ctx.reply(formatted_results)


# 错误处理
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("抱歉，我不认识这个命令。输入 `!help` 查看可用命令。")
    else:
        logger.error(f"命令错误: {traceback.format_exc()}")
        await ctx.send(f"执行命令时出错，请稍后再试。")


# 启动机器人
if __name__ == "__main__":
    try:
        bot.run(TOKEN)
    except Exception as e:
        logger.critical(f"启动机器人时出错: {traceback.format_exc()}")
