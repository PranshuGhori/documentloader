# RAG Document Loader & Agent

This is a Retrieval-Augmented Generation (RAG) project that uses **LangChain** and **xAI's Grok API** (`grok-4-fast-non-reasoning`) to intelligently load documents, process text, and answer questions.

## 🚀 Features
- Uses LangChain's abstractions for document loading and orchestration.
- Connects directly to xAI's state-of-the-art Grok models.
- Support for parsing text, notes, and local PDFs smoothly.

## 🛠️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/PranshuGhori/documentloader.git
cd documentloader
```

**2. Create a virtual environment & install requirements**
```bash
# We recommend using uv or standard pip
uv venv .venv
source .venv/bin/activate

uv pip install -r requirement.txt
```

**3. Configure Environment Variables**
Create a `.env` file in the root directory and add your xAI API key:
```env
XAI_API_KEY="your-xai-api-key-here"
```
*(Note: Your `.env` file is gitignored automatically).*

## 🏃‍♂️ Usage
Once everything is installed and your Grok API key is configured, you can run the primary script:
```bash
python main.py
```
This will initialize the LLM and run the query!
