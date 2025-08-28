#!/bin/bash
echo "🚀 启动 Funding Agent..."

# 激活环境
conda activate funding_agent_env

# 显示环境信息
echo "📍 Python: $(which python)"
echo "📍 环境: $CONDA_DEFAULT_ENV"

# 启动应用
python -m streamlit run app.py
