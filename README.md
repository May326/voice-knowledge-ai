# voice-knowledge-ai
语音对话
# 🎙️ 语音知识库AI

本地运行的语音对话AI系统，支持：
- 🗣️ 语音问答
- 📚 本地知识库
- 🤖 AI对话
- 🔊 语音播报

## 快速开始

### 在GitHub Codespaces中运行
1. 点击 `Code` → `Codespaces` → `Create codespace`
2. 等待环境初始化
3. 运行：`python app.py`
4. 打开浏览器访问提示的URL

### 本地运行
```bash
pip install -r requirements.txt
python app.py
```

## 功能说明

- **上传文档**：支持PDF、Word、TXT导入知识库
- **语音输入**：点击录音按钮提问
- **文字输入**：直接输入问题
- **知识管理**：查看和删除知识库内容

## 技术栈

- Whisper: 语音识别
- ChromaDB: 向量数据库
- Gradio: Web界面
- Edge-TTS: 语音合成
- Ollama: 本地LLM（可选）
