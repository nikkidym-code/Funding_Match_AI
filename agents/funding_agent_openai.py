from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

class FundingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Import tools
        from tools import SearchFundingTool, CheckEligibilityTool, WebSearchTool
        self.tools = [SearchFundingTool, CheckEligibilityTool, WebSearchTool]
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Use agent type that supports multi-input tools
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=2,
            max_execution_time=30,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
            early_stopping_method="force"
        )
    
    def process_query(self, query: str, company_info: dict = None) -> str:
        """Process user query"""
        try:
            # Convert to English and provide clear instructions
            if company_info:
                enhanced_query = f"""
User Request: Please help find suitable investment institutions for our company.

Company Details:
- Company Name: {company_info.get('company_name', 'Not provided')}
- Industry: {company_info.get('industry', 'Not provided')}
- Country: {company_info.get('country', 'Not provided')}
- Funding Stage: {company_info.get('stage', 'Not provided')}
- Founded Year: {company_info.get('founded_year', 'Not provided')}
- Team Size: {company_info.get('team_size', 'Not provided')}
- Annual Revenue: ${company_info.get('revenue', 0):,}
- Description: {company_info.get('description', 'Not provided')}

CRITICAL INSTRUCTIONS:
1. Use SearchFundingTool with these exact parameters:
   - industry: "{company_info.get('industry', 'AI')}"
   - country: "{company_info.get('country', 'UK')}"
   - stage: "{company_info.get('stage', 'Pre-Seed')}"
   - description: "{company_info.get('description', '')}"

2. After getting the search results, immediately provide a Final Answer in Chinese that includes:
   - List of recommended investment institutions
   - Practical advice for approaching investors
   - Next steps

3. DO NOT call any tools after SearchFundingTool
4. DO NOT say "I will provide" or "Next I will" - just provide the Final Answer immediately
5. Format the response nicely in Chinese for the user
"""
            else:
                # Handle queries without company info
                english_query = self._translate_query_intent(query)
                enhanced_query = f"""
User Request: {english_query}

INSTRUCTIONS:
1. Understand what the user is asking for
2. Use appropriate tools if needed (only once per tool type)
3. Provide a complete Final Answer in Chinese
4. Do not repeat tool calls
5. Be helpful and professional
"""
            
            print(f"Sending English prompt to agent...")
            
            response = self.agent_executor.invoke({"input": enhanced_query})
            result = response.get("output", "")
            
            # Check if we got a proper response
            if not result or len(result.strip()) < 50:
                print("Agent response too short, using fallback")
                return self._fallback_response(query, company_info)
            
            return result
            
        except Exception as e:
            print(f"Agent execution error: {str(e)}")
            return self._fallback_response(query, company_info)
    
    def _translate_query_intent(self, query: str) -> str:
        """Translate Chinese query intent to English"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["投资机构", "推荐", "融资"]):
            return "Please recommend suitable investment institutions for our company"
        elif any(keyword in query_lower for keyword in ["风险", "评估", "资格"]):
            return "Please assess our company's funding eligibility and risks"
        elif any(keyword in query_lower for keyword in ["市场", "趋势", "行业"]):
            return "Please provide market insights and industry trends"
        else:
            return f"Please help with: {query}"
    
    def _fallback_response(self, query: str, company_info: dict = None) -> str:
        """Fallback response when agent fails"""
        try:
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ["投资机构", "推荐", "funding", "investor"]):
                return self._direct_search_funding(company_info)
            elif any(keyword in query_lower for keyword in ["风险", "评估", "资格"]):
                return self._direct_check_eligibility(company_info)
            else:
                return self._direct_llm_response(query, company_info)
                
        except Exception as e:
            return f"系统处理请求时出现问题：{str(e)}"
    
    def _direct_search_funding(self, company_info: dict) -> str:
        """Direct funding search"""
        if not company_info:
            return "❌ 请先在左侧填写公司信息以获得精准推荐。"
        
        try:
            from tools.funding_rag_tool_openai import search_funding
            result = search_funding(
                industry=company_info.get('industry', 'AI'),
                country=company_info.get('country', 'UK'),
                stage=company_info.get('stage', 'Pre-Seed'),
                description=company_info.get('description', '')
            )
            
            return f"""## 🎯 为您推荐的投资机构

{result}

## 💡 融资建议

基于您的公司信息，建议您：

### 📋 准备工作
1. **完善商业计划书** - 确保逻辑清晰，数据可信
2. **财务模型** - 制作3-5年的收入和成本预测
3. **产品演示** - 准备简洁有力的产品Demo
4. **团队介绍** - 突出核心团队的专业背景

### 🤝 接触策略
1. **通过熟人介绍** - 最有效的方式
2. **参加行业活动** - 增加曝光度
3. **LinkedIn建联** - 建立专业关系
4. **准备电梯演讲** - 30秒内说清楚业务核心

### 📈 提升竞争力
1. **用户验证** - 获得更多真实用户反馈
2. **技术壁垒** - 强化核心技术优势
3. **合作伙伴** - 建立战略合作关系
4. **收入模式** - 证明商业模式的可行性

如需进一步的风险评估，请随时告诉我！"""
            
        except Exception as e:
            return f"搜索投资机构时出错：{str(e)}"
    
    def _direct_check_eligibility(self, company_info: dict) -> str:
        """Direct eligibility check"""
        if not company_info:
            return "❌ 请先在左侧填写公司信息以进行评估。"
        
        try:
            from tools.risk_predict_tool import check_eligibility
            
            assessment_info = {
                'company_age': 2025 - company_info.get('founded_year', 2023),
                'annual_revenue': company_info.get('revenue', 0),
                'team_size': company_info.get('team_size', 5),
                'funding_rounds': company_info.get('funding_rounds', 0),
                'industry': company_info.get('industry', ''),
                'description': company_info.get('description', ''),
                'has_financial_records': True,
                'has_business_plan': True
            }
            
            result = check_eligibility(assessment_info)
            return f"""## 📊 公司融资资格与风险评估

{result}

## 🚀 改进建议

### 💪 继续发挥的优势
- 保持技术创新领先地位
- 扩大用户基础和市场占有率
- 加强团队建设和人才储备

### 🔧 需要重点改进的方面
- 完善财务管理和报告体系
- 建立清晰可验证的盈利模式
- 制定详细的市场扩张计划
- 加强风险控制和合规管理

### 📋 下一步行动计划
1. **准备尽职调查材料** - 整理完整的公司文档
2. **优化商业模式** - 明确价值主张和收入来源
3. **建立投资者关系** - 寻找合适的FA或导师
4. **市场验证** - 获得更多客户和收入证明

根据评估结果，建议您优先解决风险评分较高的问题，这将显著提升融资成功率。"""
            
        except Exception as e:
            return f"风险评估时出错：{str(e)}"
    
    def _direct_llm_response(self, query: str, company_info: dict) -> str:
        """Direct LLM response"""
        try:
            context = ""
            if company_info:
                context = f"""
公司背景：
- 行业：{company_info.get('industry', '未提供')}
- 阶段：{company_info.get('stage', '未提供')}
- 描述：{company_info.get('description', '未提供')}
"""
            
            prompt = f"""
作为专业的融资顾问，请回答用户的问题：{query}

{context}

请提供实用、专业的建议，重点关注融资相关内容。
"""
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"生成回答时出错：{str(e)}"