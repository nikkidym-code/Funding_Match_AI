# app.py - 修复重复显示问题
import streamlit as st
from agents.funding_agent import SmartFundingAgent
from tools.funding_rag_tool import build_vector_store
import os
import json

# 页面配置
st.set_page_config(
    page_title="💸 FundingMatch AI",
    page_icon="💸",
    layout="wide"
)

# 初始化
if 'agent' not in st.session_state:
    st.session_state.agent = SmartFundingAgent()
    
if 'company_info' not in st.session_state:
    st.session_state.company_info = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 构建向量库（首次运行）
@st.cache_resource
def initialize_vector_store():
    data_files = [
        "data/eu_investors_uk.csv",
        "data/eu_investors_germany.csv"
    ]
    build_vector_store(data_files)

def process_button_query(query, company_info):
    """处理按钮触发的查询"""
    # 添加到聊天历史
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # 初始化向量库
    initialize_vector_store()
    
    # 获取响应
    try:
        response = st.session_state.agent.process_query(query, company_info)
    except Exception as e:
        response = f"Error processing query: {str(e)}"
    
    # 添加到聊天历史
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    return response

# 标题
st.title("💸 FundingMatch AI - Intelligent Funding Advisor")
st.markdown("---")

# 侧边栏 - 公司信息表单
with st.sidebar:
    st.header("📋 Company Information")
    
    with st.form("company_form"):
        company_name = st.text_input("Company Name *", placeholder="e.g., AI Healthcare Tech")
        industry = st.text_input("Industry *", placeholder="e.g., AI+Healthcare")
        country = st.selectbox("Country/Region *", [
  "Austria",
  "Belgium",
  "Bulgaria",
  "Croatia",
  "Cyprus",
  "Czech Republic",
  "Denmark",
  "Estonia",
  "Finland",
  "France",
  "Germany",
  "Greece",
  "Hungary",
  "Ireland",
  "Italy",
  "Latvia",
  "Lithuania",
  "Luxembourg",
  "Malta",
  "Netherlands",
  "Norway",
  "Poland",
  "Portugal",
  "Romania",
  "Slovakia",
  "Slovenia",
  "Spain",
  "Sweden",
  "Switzerland",
  "United Kingdom"
])
        stage = st.selectbox("Funding Stage *", ["Pre-Seed", "Seed", "Series A", "Series B", "Series C+"])
        founded_year = st.number_input("Founded Year *", min_value=2000, max_value=2025, value=2023)
        team_size = st.number_input("Team Size", min_value=1, max_value=1000, value=5)
        funding_amount = st.number_input("Expected Funding Amount (EUR)", min_value=0, value=0, step=100000)
        funding_rounds = st.number_input("Completed Funding Rounds", min_value=0, max_value=10, value=0)
        revenue = st.number_input("Annual Revenue (EUR)", min_value=0, value=0, step=100000)
        description = st.text_area(
            "Company Description *", 
            placeholder="Please describe your company's business, core technologies, market positioning, etc. in detail...",
            height=150
        )
        
        submitted = st.form_submit_button("Save Information and Start Conversation")
        
        if submitted:
            if all([company_name, industry, country, stage, description]):
                st.session_state.company_info = {
                    "company_name": company_name,
                    "industry": industry,
                    "country": country,
                    "stage": stage,
                    "founded_year": founded_year,
                    "team_size": team_size,
                    "funding_rounds": funding_rounds,
                    "revenue": revenue,
                    "description": description,
                    "funding_amount": funding_amount
                }
                st.success("✅ Company information has been saved!")
            else:
                st.error("Please fill in all required fields (*)")

# 主界面
if st.session_state.company_info is None:
    # 欢迎页面
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Please fill in company information on the left side first")
        st.markdown("""
        ### 🚀 Welcome to FundingMatch AI
        
        Our AI assistant can help you:
        - 🔍 **Smart Matching Investment Institutions** - Recommend the most suitable investment institutions based on your business characteristics
        - 📊 **Risk Assessment** - Comprehensive analysis of your funding readiness and potential risks
        - 🌐 **Market Insights** - Provide the latest industry dynamics and investment trends
        
        **Steps:**
        1. Fill in company basic information on the left side
        2. Click "Save Information and Start Conversation" 
        3. Chat with the AI assistant to get personalized funding advice
        """)
else:
    # 显示公司信息摘要
    with st.expander("📊 Current Company Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Company Name:** {st.session_state.company_info['company_name']}")
            st.write(f"**Industry:** {st.session_state.company_info['industry']}")
            st.write(f"**Region:** {st.session_state.company_info['country']}")
        with col2:
            st.write(f"**Funding Stage:** {st.session_state.company_info['stage']}")
            st.write(f"**Team Size:** {st.session_state.company_info['team_size']} people")
            st.write(f"**Founded Year:** {st.session_state.company_info['founded_year']}")
        with col3:
            st.write(f"**Funding Amount:** ${st.session_state.company_info['funding_amount']:,}")
            st.write(f"**Completed Funding Rounds:** {st.session_state.company_info['funding_rounds']}")
            st.write(f"**Annual Revenue:** ${st.session_state.company_info['revenue']:,}")
            
    # 快速操作按钮
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Find matching investors", key="search_button"):
            query = "Please recommend the most suitable investment institutions based on my company information"
            with st.spinner("Searching for investment institutions..."):
                process_button_query(query, st.session_state.company_info)
                st.success("✅ Recommended results have been added to the conversation below!")
    
    with col2:
        if st.button("📊 Evaluate Funding Risk", key="risk_button"):
            query = "Please evaluate the current funding risk and readiness of my company"
            with st.spinner("Evaluating risk..."):
                process_button_query(query, st.session_state.company_info)
                st.success("✅ Evaluation results have been added to the conversation below!")
    
    with col3:
        if st.button("🌐 Get Market Insights", key="market_button"):
            query = f"Please analyze the latest funding trends in the {st.session_state.company_info['industry']} industry"
            with st.spinner("Analyzing market trends..."):
                process_button_query(query, st.session_state.company_info)
                st.success("✅ Analysis results have been added to the conversation below!")
    
    # 对话界面
    st.markdown("### 💬 AI Funding Advisor Conversation")
    
    # 显示聊天历史
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # 用户输入处理
    user_input = st.chat_input("Please enter your question in English...")
    
    if user_input:
        # 显示用户消息
        with st.chat_message("user"):
            st.write(user_input)
        
        # 添加到聊天历史
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 生成AI回复
        with st.chat_message("assistant"):
            with st.spinner("🤔 AI is thinking..."):
                # 初始化向量库
                initialize_vector_store()
                
                # 获取响应
                try:
                    response = st.session_state.agent.process_query(user_input, st.session_state.company_info)
                except Exception as e:
                    response = f"Error processing query: {str(e)}"
                
                # 显示响应
                st.write(response)
                
                # 添加到聊天历史
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# 页脚
st.markdown("---")
st.markdown("💡 **Note:** You can update company information on the left side at any time to get more accurate recommendations.")