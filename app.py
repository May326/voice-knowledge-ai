import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
import PyPDF2
import docx
import whisper
import edge_tts
import asyncio
import tempfile

# ===== LLM 配置 =====
LLM_TYPE = "GEMINI"  # 可选: "GEMINI", "OPENAI", "OLLAMA"

if LLM_TYPE == "GEMINI":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')  # 使用最新的免费模型
elif LLM_TYPE == "OPENAI":
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
else:  # OLLAMA
    try:
        import ollama
    except:
        print("⚠️  请安装 ollama: pip install ollama")

class EnglishConversationAI:
    def __init__(self):
        # 初始化知识库
        self.kb_path = "./data/chroma_db"
        os.makedirs(self.kb_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.kb_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        try:
            self.collection = self.client.get_collection(
                name="knowledge_base",
                embedding_function=self.embedding_fn
            )
        except:
            self.collection = self.client.create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_fn
            )
        
        # 初始化语音识别模型（延迟加载）
        self.whisper_model = None
        
        # TTS语音配置 - 英文语音
        self.tts_voices = {
            "美式女声": "en-US-AriaNeural",
            "美式男声": "en-US-GuyNeural",
            "英式女声": "en-GB-SoniaNeural",
            "英式男声": "en-GB-RyanNeural",
        }
        self.current_voice = "en-US-AriaNeural"
        
        # 对话历史
        self.conversation_history = []
    
    def load_whisper(self):
        """延迟加载Whisper模型"""
        if self.whisper_model is None:
            print("正在加载Whisper模型...")
            self.whisper_model = whisper.load_model("base")
            print("Whisper模型加载完成")
        return self.whisper_model
    
    def transcribe_audio(self, audio_path):
        """语音转文字 - 英文识别"""
        if not audio_path:
            return ""
        
        try:
            model = self.load_whisper()
            # 指定为英文识别
            result = model.transcribe(audio_path, language="en")
            return result["text"]
        except Exception as e:
            return f"Speech recognition error: {str(e)}"
    
    async def text_to_speech_async(self, text, voice=None):
        """文字转语音（异步）- 英文"""
        if not text or text.startswith("Error") or text.startswith("❌"):
            return None
        
        try:
            # 限制语音长度
            if len(text) > 800:
                text = text[:800] + "..."
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            output_path = temp_file.name
            temp_file.close()
            
            voice_to_use = voice if voice else self.current_voice
            communicate = edge_tts.Communicate(text, voice_to_use)
            await communicate.save(output_path)
            
            return output_path
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None
    
    def text_to_speech(self, text, voice=None):
        """文字转语音（同步封装）"""
        try:
            return asyncio.run(self.text_to_speech_async(text, voice))
        except Exception as e:
            print(f"Speech synthesis error: {str(e)}")
            return None
    
    def generate_conversation_response(self, user_input, context_docs, mode="conversation", difficulty="intermediate"):
        """生成对话回复 - 针对英语练习优化"""
        
        # 组合知识库上下文
        context = ""
        if context_docs:
            context = "\n\n".join([f"[Reference {i+1}]\n{doc}" for i, doc in enumerate(context_docs)])
        
        # 根据模式调整提示词
        system_prompts = {
            "conversation": """You are a friendly English conversation partner helping someone practice English. 
Based on the knowledge base content, engage in natural conversation. 
- Speak naturally and encouragingly
- Use appropriate vocabulary for their level
- Ask follow-up questions to keep conversation flowing
- Correct major errors gently
- Be supportive and patient""",
            
            "roleplay": """You are helping someone practice English through roleplay scenarios.
Based on the knowledge base content (which may describe situations, dialogues, or scenarios):
- Stay in character for the scenario
- Use realistic, situational language
- Provide natural responses as if in a real situation
- Help them practice practical English for real-world use""",
            
            "discussion": """You are an English tutor facilitating topic discussions.
Based on the knowledge base content:
- Discuss the topic in depth
- Ask thought-provoking questions
- Encourage the student to express opinions
- Introduce relevant vocabulary and expressions
- Provide examples and explanations when needed"""
        }
        
        difficulty_notes = {
            "beginner": "Use simple vocabulary and short sentences. Speak slowly and clearly.",
            "intermediate": "Use everyday vocabulary with some advanced words. Speak at normal pace.",
            "advanced": "Use sophisticated vocabulary and complex sentences. Discuss abstract concepts."
        }
        
        # 构建对话历史
        history_text = ""
        if self.conversation_history:
            history_text = "\n\nConversation History:\n"
            for entry in self.conversation_history[-6:]:  # 只保留最近3轮对话
                history_text += f"Student: {entry['user']}\nTeacher: {entry['assistant']}\n\n"
        
        prompt = f"""{system_prompts.get(mode, system_prompts['conversation'])}

Difficulty Level: {difficulty}
Note: {difficulty_notes.get(difficulty, difficulty_notes['intermediate'])}

Knowledge Base Content:
{context if context else "No specific reference material. Engage in general conversation."}
{history_text}
Student: {user_input}

Teacher (respond in English, naturally and helpfully):"""

        try:
            if LLM_TYPE == "GEMINI":
                response = gemini_model.generate_content(prompt)
                return response.text
            
            elif LLM_TYPE == "OPENAI":
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompts.get(mode, system_prompts['conversation'])},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=300
                )
                return response.choices[0].message.content
            
            else:  # OLLAMA
                response = ollama.chat(
                    model='qwen2:7b',
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return response['message']['content']
        
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "API key" in error_msg:
                return f"❌ API Key not configured.\n\nPlease set: export GOOGLE_API_KEY='your-key'"
            else:
                return f"❌ LLM Error: {error_msg}"
    
    def extract_text_from_file(self, file_path):
        """从文件中提取文本"""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_ext == '.pdf':
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            elif file_ext in ['.docx', '.doc']:
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            
            else:
                return None
        except Exception as e:
            return f"Error: {str(e)}"
    
    def add_document(self, file_path, chunk_size=500):
        """添加学习材料到知识库"""
        if not file_path:
            return "❌ Please upload a file first"
        
        text = self.extract_text_from_file(file_path)
        if not text:
            return "❌ Unsupported file format or empty file"
        
        if text.startswith("Error"):
            return text
        
        # 分块
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        
        filename = Path(file_path).name
        current_count = self.collection.count()
        
        ids = [f"{filename}_chunk_{i+current_count}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        return f"✅ Successfully added {len(chunks)} content blocks\nFile: {filename}"
    
    def practice_conversation(self, user_input, mode, difficulty, voice, n_results=3):
        """进行对话练习"""
        if not user_input.strip():
            return "Please say something", None, ""
        
        # 检索相关知识库内容
        context_docs = []
        if self.collection.count() > 0:
            results = self.collection.query(
                query_texts=[user_input],
                n_results=min(n_results, self.collection.count())
            )
            if results['documents'][0]:
                context_docs = results['documents'][0]
        
        # 生成回复
        response = self.generate_conversation_response(
            user_input, 
            context_docs, 
            mode=mode, 
            difficulty=difficulty
        )
        
        # 记录对话历史
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        # 更新语音
        self.current_voice = self.tts_voices.get(voice, "en-US-AriaNeural")
        
        # 生成语音
        audio_path = self.text_to_speech(response)
        
        # 构建显示的对话历史
        chat_display = ""
        for entry in self.conversation_history[-10:]:  # 显示最近10轮
            chat_display += f"**🧑 You:** {entry['user']}\n\n"
            chat_display += f"**🤖 Teacher:** {entry['assistant']}\n\n"
            chat_display += "---\n\n"
        
        return chat_display, audio_path, ""
    
    def practice_with_voice(self, audio, mode, difficulty, voice, n_results=3):
        """语音对话练习"""
        if audio is None:
            return "Please record your voice", None, "", ""
        
        # 语音转文字
        user_input = self.transcribe_audio(audio)
        
        if not user_input or user_input.startswith("Error"):
            return user_input, None, "", ""
        
        # 进行对话
        chat_display, audio_path, _ = self.practice_conversation(
            user_input, mode, difficulty, voice, n_results
        )
        
        return chat_display, audio_path, user_input, ""
    
    def reset_conversation(self):
        """重置对话"""
        self.conversation_history = []
        return "✅ Conversation reset. Ready for a new practice session!"
    
    def get_stats(self):
        """获取知识库统计"""
        count = self.collection.count()
        if count == 0:
            return "Knowledge base is empty. Upload some learning materials to get started!"
        
        results = self.collection.get()
        sources = set([m.get('source', 'Unknown') for m in results['metadatas']])
        
        stats = f"📊 Knowledge Base Statistics\n\n"
        stats += f"- Total content blocks: {count}\n"
        stats += f"- Source files: {len(sources)}\n"
        stats += f"- Files:\n"
        for source in sources:
            stats += f"  • {source}\n"
        
        return stats
    
    def clear_database(self):
        """清空知识库"""
        try:
            self.client.delete_collection(name="knowledge_base")
            self.collection = self.client.create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_fn
            )
            return "✅ Knowledge base cleared"
        except Exception as e:
            return f"❌ Clear failed: {str(e)}"

# 初始化AI
ai = EnglishConversationAI()

# 创建Gradio界面
with gr.Blocks(title="AI 英语口语练习助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ AI 英语口语练习助手
    
    基于你的学习材料，与 AI 进行英语对话练习！
    
    🤖 由 **Google Gemini 1.5 Flash** 驱动
    """)
    
    with gr.Tabs():
        # Tab 1: 上传学习材料
        with gr.Tab("📤 上传学习材料"):
            gr.Markdown("""
            上传英语学习材料，例如：
            - 教材对话
            - 话题文章
            - 场景对话
            - 词汇列表
            """)
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="选择文件",
                        file_types=[".txt", ".pdf", ".docx", ".doc"]
                    )
                    chunk_size = gr.Slider(
                        minimum=200,
                        maximum=1000,
                        value=500,
                        step=50,
                        label="文本块大小"
                    )
                    upload_btn = gr.Button("📁 添加到知识库", variant="primary")
                
                with gr.Column():
                    upload_output = gr.Textbox(
                        label="上传结果",
                        lines=5
                    )
            
            upload_btn.click(
                fn=lambda f, c: ai.add_document(f.name if f else None, c),
                inputs=[file_input, chunk_size],
                outputs=upload_output
            )
        
        # Tab 2: 语音对话练习
        with gr.Tab("🎤 语音对话"):
            gr.Markdown("### 点击麦克风按钮，用英语说出你想说的话")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎤 录制语音"
                    )
                    
                    mode_voice = gr.Radio(
                        choices=["conversation", "roleplay", "discussion"],
                        value="conversation",
                        label="练习模式",
                        info="对话：自由聊天 | 角色扮演：场景练习 | 讨论：话题讨论"
                    )
                    
                    difficulty_voice = gr.Radio(
                        choices=["beginner", "intermediate", "advanced"],
                        value="intermediate",
                        label="难度级别"
                    )
                    
                    voice_select_voice = gr.Radio(
                        choices=list(ai.tts_voices.keys()),
                        value="美式女声",
                        label="老师语音"
                    )
                    
                    voice_btn = gr.Button("🗣️ 开始练习", variant="primary", size="lg")
                    reset_btn_voice = gr.Button("🔄 重置对话", variant="secondary")
                
                with gr.Column(scale=2):
                    recognized_voice = gr.Textbox(
                        label="📝 你说的内容",
                        lines=2
                    )
                    conversation_display_voice = gr.Textbox(
                        label="💬 对话历史",
                        lines=12
                    )
                    audio_output_voice = gr.Audio(
                        label="🔊 老师的回复",
                        autoplay=True
                    )
            
            voice_btn.click(
                fn=ai.practice_with_voice,
                inputs=[audio_input, mode_voice, difficulty_voice, voice_select_voice],
                outputs=[conversation_display_voice, audio_output_voice, recognized_voice, audio_input]
            )
            
            reset_btn_voice.click(
                fn=ai.reset_conversation,
                outputs=conversation_display_voice
            )
        
        # Tab 3: 文字对话练习
        with gr.Tab("💬 文字对话"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="用英语输入你的消息",
                        placeholder="Hello! I'd like to practice English conversation...",
                        lines=3
                    )
                    
                    mode_text = gr.Radio(
                        choices=["conversation", "roleplay", "discussion"],
                        value="conversation",
                        label="练习模式"
                    )
                    
                    difficulty_text = gr.Radio(
                        choices=["beginner", "intermediate", "advanced"],
                        value="intermediate",
                        label="难度级别"
                    )
                    
                    voice_select_text = gr.Radio(
                        choices=list(ai.tts_voices.keys()),
                        value="美式女声",
                        label="老师语音"
                    )
                    
                    text_btn = gr.Button("💬 发送", variant="primary", size="lg")
                    reset_btn_text = gr.Button("🔄 重置对话", variant="secondary")
                
                with gr.Column(scale=2):
                    conversation_display_text = gr.Textbox(
                        label="💬 对话历史",
                        lines=15
                    )
                    audio_output_text = gr.Audio(
                        label="🔊 老师的回复",
                        autoplay=True
                    )
            
            gr.Examples(
                examples=[
                    ["Hello! How are you today?"],
                    ["Can you help me practice ordering food at a restaurant?"],
                    ["What do you think about artificial intelligence?"],
                    ["I'd like to discuss environmental issues."],
                ],
                inputs=text_input
            )
            
            text_btn.click(
                fn=ai.practice_conversation,
                inputs=[text_input, mode_text, difficulty_text, voice_select_text],
                outputs=[conversation_display_text, audio_output_text, text_input]
            )
            
            reset_btn_text.click(
                fn=ai.reset_conversation,
                outputs=conversation_display_text
            )
        
        # Tab 4: 知识库管理
        with gr.Tab("⚙️ 知识库管理"):
            with gr.Row():
                stats_btn = gr.Button("📊 查看统计")
                clear_btn = gr.Button("🗑️ 清空知识库", variant="stop")
            
            stats_output = gr.Textbox(
                label="统计信息",
                lines=10
            )
            
            stats_btn.click(
                fn=ai.get_stats,
                outputs=stats_output
            )
            
            clear_btn.click(
                fn=ai.clear_database,
                outputs=stats_output
            )
    
    gr.Markdown("""
    ---
    ### 💡 使用方法
    
    1. **上传学习材料**：添加英语学习资料（对话、文章、场景等）
    2. **选择模式**：
       - 🗣️ **对话 (Conversation)**：自然的自由对话
       - 🎭 **角色扮演 (Roleplay)**：练习真实场景（餐厅、购物等）
       - 💭 **讨论 (Discussion)**：深入讨论话题
    3. **选择难度**：初级 (Beginner)、中级 (Intermediate)、高级 (Advanced)
    4. **开始练习**：用英语说话或打字，AI 会自然回复
    
    ### 🎯 功能特点
    
    - ✅ 基于你的学习材料进行自然英语对话
    - ✅ 多个难度级别可选
    - ✅ 不同的练习模式
    - ✅ 语音识别和合成
    - ✅ 对话历史记录
    - ✅ 上下文感知回复
    
    ### 🔧 配置说明
    
    ```bash
    export GOOGLE_API_KEY="你的key"
    ```
    在这里免费获取：https://makersuite.google.com/app/apikey
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )