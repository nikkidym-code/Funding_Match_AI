# FundingMatch AI - Intelligent Investment Matching System

## 🚀 Project Overview

FundingMatch AI is an AI-powered intelligent investment matching system that uses multi-dimensional analysis and machine learning technology to recommend the most suitable investment institutions for startups.

## ✨ Core Features

- **Intelligent Investment Matching**: Semantic search-based investment institution recommendations
- **Risk Assessment**: Machine learning-driven investment success probability prediction
- **Market Intelligence**: Real-time market trends and industry analysis
- **Smart Conversation**: Natural language interaction for investment consultation

## 🏗️ Technical Architecture

- **AI Agent**: Intelligent agent system based on LangChain
- **Vector Database**: ChromaDB for semantic search implementation
- **Machine Learning**: Integration of multiple ML models for risk assessment
- **Web Interface**: User-friendly interface built with Streamlit
- **LLM Integration**: Support for OpenAI and DeepSeek models

## 📋 System Requirements

- Python 3.8+
- 8GB+ RAM
- Stable internet connection

## 🛠️ Installation Steps

### 1. Clone the Project
```bash
git clone https://github.com/yourusername/Funding_Agent_V3.git
cd Funding_Agent_V3
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables Configuration

#### Method 1: Create .env File (Recommended)
Create a `.env` file in the project root directory:

```bash
# LLM Provider Configuration
LLM_PROVIDER=openai  # or deepseek
LLM_MODEL=gpt-4o-mini  # OpenAI model name

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# DeepSeek Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Other API Configuration
TAVILY_API_KEY=your_tavily_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

#### Method 2: System Environment Variables
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key_here
export LLM_MODEL=gpt-4o-mini
```

### 5. Run the Application
```bash
streamlit run app.py
```

## 🔧 Configuration Guide

### LLM Provider Switching

The system supports two LLM providers:

1. **OpenAI** (Recommended for Production)
   - Models: GPT-4o-mini, GPT-4
   - Advantages: Stable performance, powerful features
   - Use Case: Production deployment and advanced features

2. **DeepSeek** (Recommended for Development/Testing)
   - Models: deepseek-chat
   - Advantages: Lower cost, suitable for development testing
   - Use Case: Feature validation and cost control

### Switching Methods

```python
# Specify in code
agent = SmartFundingAgent(llm_provider="openai", llm_model="gpt-4o-mini")

# Or through environment variables
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
```

## 📁 Project Structure

```
Funding_Agent_V3/
├── app.py                 # Streamlit main application
├── agents/               # AI agent modules
│   ├── __init__.py
│   └── funding_agent.py  # Core intelligent agent
├── tools/                # Tool modules
│   ├── funding_rag_tool.py    # Investment matching tool
│   ├── risk_predict_tool.py   # Risk assessment tool
│   └── web_search_tool.py     # Web search tool
├── data/                 # Data files
│   └── eu_investors_all_countries.csv
├── models/               # Machine learning models
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (not committed to Git)
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

## 🚀 Deploy to Streamlit Community Cloud

### 1. Prepare for Deployment
Ensure your code has been pushed to GitHub repository and contains:
- `requirements.txt`
- `app.py` as the main file
- No `.env` file (excluded via .gitignore)

### 2. Deployment Steps
1. Visit [Streamlit Community Cloud](https://share.streamlit.io/)
2. Login with your GitHub account
3. Select your repository
4. Configure deployment settings:
   - **Main file path**: `app.py`
   - **Python version**: 3.8+
5. Add environment variables in "Advanced settings":
   - `LLM_PROVIDER`
   - `OPENAI_API_KEY`
   - `LLM_MODEL`
   - Other necessary API keys

### 3. Environment Variables Setup
In Streamlit Community Cloud deployment settings, add all necessary environment variables to ensure consistency with your local `.env` file configuration.

## 🔒 Security Considerations

- **Never** commit `.env` files to Git repositories
- Regularly rotate API keys
- Use strong passwords and secure API keys in production environments
- Monitor API usage and costs

## 🐛 Frequently Asked Questions

### Q: Why does the system display "Typical Investment: N/A"?
A: Check if the column names in the data file match the metadata keys in the code.

### Q: How to resolve LLM model switching issues?
A: Ensure environment variables are correctly set and verify API key validity.

### Q: What if the system enters an infinite loop?
A: Check Streamlit session state management and ensure button logic is correct.

## 📊 Performance Optimization

- Enable vector database caching
- Use appropriate LLM models (DeepSeek for development, OpenAI for production)
- Optimize query strategies and tool calling order

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

