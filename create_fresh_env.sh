#!/bin/bash
# create_fresh_env.sh - 创建全新的兼容环境

echo "🧹 创建全新的兼容环境..."

ENV_NAME="funding_agent_env"

# 1. 确保删除旧环境
echo "🗑️ 清理旧环境..."
conda deactivate 2>/dev/null || true
conda env remove -n risk_assessment -y 2>/dev/null || true
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# 2. 检查 Python 版本选择
echo "📋 选择 Python 版本:"
echo "1. Python 3.10 (推荐，支持最新 Streamlit)"
echo "2. Python 3.9 (兼容性好，但 Streamlit 功能受限)"
read -p "请选择 (1-2): " py_choice

if [ "$py_choice" = "1" ]; then
    PYTHON_VERSION="3.10"
    STREAMLIT_VERSION=">=1.28.0"
    echo "✅ 选择 Python 3.10 + 最新 Streamlit"
else
    PYTHON_VERSION="3.9"
    STREAMLIT_VERSION="==1.12.0"
    echo "✅ 选择 Python 3.9 + 兼容 Streamlit"
fi

# 3. 创建新环境
echo "📦 创建 Python $PYTHON_VERSION 环境..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 4. 激活环境
echo "🔌 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 5. 更新 pip
pip install --upgrade pip

# 6. 安装包（分批安装避免冲突）
echo "📦 安装 Web 框架..."
if [ "$py_choice" = "1" ]; then
    pip install "streamlit>=1.28.0"
else
    pip install "streamlit==1.12.0" "altair==4.2.2" "protobuf<=3.20.3"
fi

echo "📦 安装数据处理包..."
pip install pandas numpy scikit-learn joblib matplotlib

echo "📦 安装 AI 和 LangChain..."
pip install langchain langchain-community langchain-openai openai

echo "📦 安装工具包..."
pip install python-dotenv pydantic

echo "📦 安装机器学习库..."
# 先尝试安装 XGBoost
if conda install -c conda-forge xgboost -y; then
    echo "✅ XGBoost 安装成功"
else
    echo "⚠️ XGBoost 安装失败，安装替代品..."
    pip install lightgbm catboost
fi

echo "📦 安装可选包..."
pip install chromadb faiss-cpu tavily-python shap langchain_deepseek || echo "⚠️ 部分可选包安装失败"

# 7. 验证安装
echo ""
echo "🧪 验证环境..."
python << 'EOF'
import sys
print(f"Python: {sys.version}")
print(f"Python 路径: {sys.executable}")
print()

# 测试核心包
packages = [
    ('streamlit', 'Streamlit'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('sklearn', 'Scikit-learn'),
    ('langchain', 'LangChain'),
    ('openai', 'OpenAI'),
    ('joblib', 'Joblib')
]

print("📦 核心包状态:")
all_good = True
for pkg, name in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'installed')
        print(f"  ✅ {name}: {version}")
    except ImportError:
        print(f"  ❌ {name}: 未安装")
        all_good = False

# 测试机器学习包
ml_packages = [('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM'), ('catboost', 'CatBoost')]
print("\n🤖 机器学习包:")
ml_available = 0
for pkg, name in ml_packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'installed')
        print(f"  ✅ {name}: {version}")
        ml_available += 1
    except ImportError:
        print(f"  ❌ {name}: 未安装")

if ml_available == 0:
    print("  ⚠️ 没有可用的梯度提升库")
    all_good = False

# 测试 Streamlit 功能
print("\n🌐 Streamlit 功能测试:")
try:
    import streamlit as st
    features = ['chat_input', 'cache_resource', 'chat_message']
    for feature in features:
        if hasattr(st, feature):
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature}")
except Exception as e:
    print(f"  ❌ Streamlit 测试失败: {e}")
    all_good = False

if all_good:
    print("\n🎉 环境设置完成！所有核心功能可用")
else:
    print("\n⚠️ 环境有一些问题，但基础功能可用")

print(f"\n📋 环境信息:")
print(f"  环境名: {sys.prefix.split('/')[-1]}")
print(f"  启动命令: conda activate {sys.prefix.split('/')[-1]}")
EOF

# 8. 创建兼容性补丁（如果是 Python 3.9）
if [ "$py_choice" = "2" ]; then
    echo ""
    echo "🔧 为 Python 3.9 创建 Streamlit 兼容性补丁..."
    
    cat > streamlit_compat.py << 'EOF'
# streamlit_compat.py - Streamlit 1.12.0 兼容性补丁
import streamlit as st

# 添加缺失的功能
if not hasattr(st, 'cache_resource'):
    def cache_resource(func):
        if hasattr(st, 'cache'):
            return st.cache(allow_output_mutation=True)(func)
        return func
    st.cache_resource = cache_resource

if not hasattr(st, 'chat_input'):
    def chat_input(placeholder="Type a message..."):
        if 'chat_input_key' not in st.session_state:
            st.session_state.chat_input_key = 0
        
        with st.form(f"chat_form_{st.session_state.chat_input_key}", clear_on_submit=True):
            user_input = st.text_area(placeholder, height=80)
            submitted = st.form_submit_button("发送 📤")
            
            if submitted and user_input.strip():
                st.session_state.chat_input_key += 1
                return user_input.strip()
        return None
    st.chat_input = chat_input

if not hasattr(st, 'chat_message'):
    class ChatMessage:
        def __init__(self, role):
            self.role = role
        def __enter__(self):
            if self.role == "user":
                st.markdown("### 👤 您")
            else:
                st.markdown("### 🤖 智能助手")
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            st.markdown("---")
    
    def chat_message(role):
        return ChatMessage(role)
    st.chat_message = chat_message

print("✅ Streamlit 兼容性补丁已加载")
EOF

    echo "💡 如果使用 Python 3.9，在 app.py 开头添加："
    echo "import streamlit_compat"
fi

# 9. 创建启动脚本
cat > start_funding_agent.sh << EOF
#!/bin/bash
echo "🚀 启动 Funding Agent..."

# 激活环境
conda activate $ENV_NAME

# 显示环境信息
echo "📍 Python: \$(which python)"
echo "📍 环境: \$CONDA_DEFAULT_ENV"

# 启动应用
python -m streamlit run app.py
EOF

chmod +x start_funding_agent.sh

echo ""
echo "🎉 全新环境创建完成！"
echo ""
echo "📋 使用说明:"
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 启动应用: python -m streamlit run app.py"
echo "3. 或使用脚本: ./start_funding_agent.sh"
echo ""
if [ "$py_choice" = "2" ]; then
    echo "💡 Python 3.9 用户注意："
    echo "   在 app.py 开头添加: import streamlit_compat"
fi