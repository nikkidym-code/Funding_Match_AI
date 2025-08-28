from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_deepseek import ChatDeepSeek
import os
import json
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

class ConversationState(Enum):
    """Conversation state enumeration"""
    INITIAL = "initial"
    COMPANY_INFO_GATHERING = "company_info_gathering"
    FUNDING_SEARCH = "funding_search"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_ANALYSIS = "market_analysis"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    RECOMMENDATION_REFINEMENT = "recommendation_refinement"

class ConversationContext:
    """Enhanced conversation context management"""
    def __init__(self):
        self.current_state = ConversationState.INITIAL
        self.conversation_history: List[Dict] = []
        self.company_info: Dict = {}
        self.last_action_results: Dict = {}
        self.pending_clarifications: List[str] = []
        self.user_preferences: Dict = {}
        self.session_start_time = datetime.now()
        
    def add_interaction(self, user_query: str, analysis_result: dict, execution_plan: dict, final_result: str):
        """Add enhanced interaction record"""
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user_query': user_query,
            'analysis_result': analysis_result,
            'execution_plan': execution_plan,
            'final_result': final_result[:200] + "..." if len(final_result) > 200 else final_result,
            'state': self.current_state.value
        })
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get recent n rounds of conversation context"""
        recent = self.conversation_history[-n:] if len(self.conversation_history) >= n else self.conversation_history
        context_parts = []
        for interaction in recent:
            context_parts.append(f"User: {interaction['user_query']}")
            context_parts.append(f"Tools Used: {self._extract_tools_from_plan(interaction.get('execution_plan', {}))}")
            context_parts.append(f"State: {interaction['state']}")
        return "\n".join(context_parts)
    
    def _extract_tools_from_plan(self, execution_plan: dict) -> str:
        """Extract tool names from execution plan"""
        if not execution_plan or 'execution_steps' not in execution_plan:
            return "none"
        tools = [step.get('tool_name', 'unknown') for step in execution_plan['execution_steps']]
        return ", ".join(tools)
    
    def has_complete_company_info(self) -> bool:
        """Check if company information is complete"""
        required_fields = ['industry', 'stage', 'country']
        return all(field in self.company_info and self.company_info[field] for field in required_fields)
    
    def get_context_summary(self) -> str:
        """Get comprehensive context summary"""
        return f"""
Current State: {self.current_state.value}
Company Info Complete: {'Yes' if self.has_complete_company_info() else 'No'}
Company Details: {json.dumps(self.company_info, indent=2)}
Recent Interactions: {len(self.conversation_history)}
Available Results: {list(self.last_action_results.keys())}
Session Duration: {datetime.now() - self.session_start_time}
"""

class ToolRegistry:
    """Tool registry for dynamic tool management"""
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools with detailed descriptions"""
        self.tools = {
            "search_funding": {
                "description": "Search investment firms database to find suitable investors",
                "function": "search_funding_tool",
                "input_requirements": "Company information (industry, stage, country, etc.)",
                "output_type": "Investment firm recommendations with analysis",
                "use_cases": [
                    "Finding investment firms for funding",
                    "Getting investor recommendations",
                    "Matching company profile with investors"
                ],
                "data_source": "Internal investment database"
            },
            "web_search": {
                "description": "Real-time web search for current information",
                "function": "web_search_tool", 
                "input_requirements": "Search query or keywords",
                "output_type": "Web search results and summaries",
                "use_cases": [
                    "Getting details about specific companies",
                    "Finding latest market information", 
                    "Researching specific investment firms",
                    "Getting real-time industry news"
                ],
                "data_source": "Real-time web search"
            },
            "risk_assessment": {
                "description": "Assess company funding readiness and success probability",
                "function": "risk_assessment_tool",
                "input_requirements": "Detailed company information (team, finances, product)",
                "output_type": "Risk assessment report with recommendations",
                "use_cases": [
                    "Evaluating funding readiness",
                    "Assessing investment attractiveness",
                    "Getting improvement recommendations"
                ],
                "data_source": "Risk assessment models and historical data"
            },
            "market_analysis": {
                "description": "Analyze industry trends and market opportunities",
                "function": "market_analysis_tool",
                "input_requirements": "Industry, region, timeframe parameters",
                "output_type": "Market analysis report with trends",
                "use_cases": [
                    "Understanding market trends",
                    "Analyzing funding environment",
                    "Competitive landscape research"
                ],
                "data_source": "Market data and trend analysis"
            }
        }
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for LLM"""
        descriptions = []
        for tool_name, tool_info in self.tools.items():
            desc = f"""
🔧 **{tool_name}**:
- Description: {tool_info['description']}
- Input: {tool_info['input_requirements']}
- Output: {tool_info['output_type']}
- Use Cases: {', '.join(tool_info['use_cases'])}
- Data Source: {tool_info['data_source']}
"""
            descriptions.append(desc)
        return "\n".join(descriptions)

class SmartFundingAgent:
    """Smart Tool Composer - Intelligent Funding Agent"""
    
    def __init__(self, llm_provider: str = None, llm_model: str = None):
        """Initialize agent with configurable LLM provider.

        Args:
            llm_provider: "deepseek" or "openai". If None, reads from env LLM_PROVIDER (default: deepseek).
            llm_model: Model name for chosen provider. If None, reads from env LLM_MODEL.
        """

        # Initialize LLM with provider selection (DeepSeek | OpenAI)
        provider = (llm_provider or os.getenv("LLM_PROVIDER", "deepseek")).strip().lower()
        model_name = llm_model or os.getenv("LLM_MODEL")

        if provider == "openai":
            try:
                # Lazy import to avoid hard dependency at module import time
                from langchain_openai import ChatOpenAI
            except Exception as _:
                raise ImportError("ChatOpenAI not available. Please install langchain-openai >= 0.1.0")

            # Default OpenAI model if not provided
            if not model_name:
                # Compact, cost-effective default suitable for tool orchestration
                model_name = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

            # Prefer environment variables for credentials/config
            openai_base_url = os.getenv("OPENAI_BASE_URL")

            # Some versions accept base_url, others read from env; handle both safely
            try:
                if openai_base_url:
                    self.llm = ChatOpenAI(model=model_name, temperature=0, base_url=openai_base_url)
                else:
                    self.llm = ChatOpenAI(model=model_name, temperature=0)
            except TypeError:
                # Fallback if base_url is not a valid parameter in this version
                self.llm = ChatOpenAI(model=model_name, temperature=0)

            print(f"🤖 Using LLM provider: OpenAI | model={model_name}")

        else:
            # Default to DeepSeek
            deepseek_model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            self.llm = ChatDeepSeek(
                model=deepseek_model_name,
                temperature=0,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                api_key=os.getenv("DEEPSEEK_API_KEY")
            )
            # print(f"🤖 Using LLM provider: DeepSeek | model={deepseek_model_name}")

        # Import tools
        try:
            from tools import SearchFundingTool, CheckEligibilityTool, WebSearchTool
            self.tools = [SearchFundingTool, CheckEligibilityTool, WebSearchTool]
        except ImportError:
            print("⚠️ Warning: Could not import tools. Some functionality may be limited.")
            self.tools = []

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # Initialize enhanced components
        self.context = ConversationContext()
        self.tool_registry = ToolRegistry()
        
        print("🚀 Smart Funding Agent initialized with intelligent tool composition")

    def process_query(self, query: str, company_info: dict = None) -> str:
        """
        Main query processing using intelligent tool composition
        Maintains backward compatibility while adding smart features
        """
        try:
            # Update company information
            if company_info:
                self.context.company_info.update(company_info)
            
            print(f"🧠 Processing query: {query}")
            print(f"📍 Current state: {self.context.current_state.value}")
            
            # 🧠 Step 1: Intelligent need analysis
            need_analysis = self._analyze_user_need(query)
            print(f"📊 Need Analysis: {need_analysis['need_type']}")
            
            # 📋 Step 2: Generate execution plan
            execution_plan = self._generate_execution_plan(need_analysis)
            print(f"🎯 Execution Plan: {execution_plan['execution_strategy']['approach']}")
            
            # ⚡ Step 3: Execute and integrate
            result = self._execute_and_integrate(query, execution_plan, need_analysis)
            
            # 📝 Record interaction
            self.context.add_interaction(query, need_analysis, execution_plan, result)

            return result

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            return f"I apologize, an error occurred while processing your request: {str(e)}"

    def _analyze_user_need(self, query: str) -> dict:
        """Step 1: Intelligent user need analysis"""
        
        context_summary = self.context.get_context_summary()
        
        analysis_prompt = f"""
You are an expert need analysis specialist for investment consulting. Analyze the user's true needs comprehensively.

【User Query】: "{query}"

【Context】:
{context_summary}

【Recent Conversation】:
{self.context.get_recent_context(3)}

Analyze the user's needs from these dimensions:

1. 🎯 **Need Type Analysis**:
   - information_seeking: User wants to learn specific information
   - decision_support: User needs advice/recommendations  
   - problem_solving: User has a specific problem to solve
   - clarification: User wants to understand previous results better

2. 🔍 **Information Requirements**:
   - data_freshness: real_time|recent|historical
   - data_type: quantitative|qualitative|mixed
   - detail_level: brief|moderate|comprehensive

3. 📝 **Entity Recognition**:
   - Extract specific company/firm names mentioned
   - Identify industry, location, stage keywords
   - Detect if user references previous results

4. 🎭 **Intent Inference**:
   - Is user satisfied with previous results?
   - Does user want to drill down into specifics?
   - Is user looking for alternatives/refinements?

5. ⚡ **Urgency Assessment**:
   - low: General inquiry, no time pressure
   - medium: Important for decision making
   - high: Critical immediate need

Return JSON format:
{{
    "need_type": {{
        "primary": "information_seeking|decision_support|problem_solving|clarification",
        "secondary": ["other applicable types"]
    }},
    "information_requirements": {{
        "data_freshness": "real_time|recent|historical",
        "data_type": "quantitative|qualitative|mixed", 
        "detail_level": "brief|moderate|comprehensive"
    }},
    "mentioned_entities": {{
        "companies": ["company names"],
        "industries": ["industry types"],
        "locations": ["geographic regions"],
        "stages": ["funding stages"]
    }},
    "user_context": {{
        "satisfaction_with_previous": "satisfied|neutral|dissatisfied|no_previous",
        "follow_up_type": "clarification|deep_dive|alternative|new_topic",
        "references_previous_results": true/false
    }},
    "urgency": "low|medium|high",
    "reasoning": "Detailed analysis of why you made these assessments"
}}

Focus on understanding what the user REALLY wants, not just surface keywords.
"""
        
        try:
            response = self.llm.invoke(analysis_prompt)
            content = self._clean_json_response(response.content)
            result = json.loads(content)
            
            # Validate required fields
            if 'need_type' not in result:
                raise ValueError("Missing need_type in analysis")
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Need analysis parsing error: {e}")
            return self._fallback_need_analysis(query)

    def _generate_execution_plan(self, need_analysis: dict) -> dict:
        """Step 2: Generate intelligent execution plan"""
        
        available_tools = self.tool_registry.get_tool_descriptions()
        
        planning_prompt = f"""
You are an expert execution planner. Design the optimal strategy to fulfill the user's needs.

【Need Analysis】:
{json.dumps(need_analysis, indent=2, ensure_ascii=False)}

【Available Tools】:
{available_tools}

【Context State】:
- Current state: {self.context.current_state.value}
- Company info complete: {self.context.has_complete_company_info()}
- Previous results available: {list(self.context.last_action_results.keys())}

Design Principles:
🎯 **Efficiency**: Minimum steps for maximum value
🔗 **Synergy**: Tools should complement each other
📊 **Quality**: Ensure accurate and complete information
⚡ **Responsiveness**: Match urgency level

Tool Selection Guidelines:
- **search_funding**: When user needs investment firm recommendations
- **web_search**: When user asks about specific companies/entities OR needs real-time info
- **risk_assessment**: When user wants to evaluate readiness/probability
- **market_analysis**: When user needs industry/market insights

Execution Strategies:
- **single_tool**: One clear, specific need
- **sequential_tools**: Information from one tool feeds into another
- **parallel_tools**: Multiple independent analyses needed
- **hybrid**: Combination of sequential and parallel

Return JSON:
{{
    "execution_strategy": {{
        "approach": "single_tool|sequential_tools|parallel_tools|hybrid",
        "rationale": "Why this approach is optimal"
    }},
    "execution_steps": [
        {{
            "step_id": 1,
            "tool_name": "tool_name",
            "purpose": "What this step accomplishes",
            "input_query": "Optimized query for this tool",
            "depends_on": null or ["previous_step_ids"],
            "priority": "high|medium|low"
        }}
    ],
    "integration_strategy": {{
        "method": "direct_presentation|synthesis|comparison|narrative",
        "focus_points": ["Key aspects to highlight"],
        "format": "Recommended response format"
    }},
    "expected_outcome": "What the user will receive"
}}

Consider the user's actual needs, not just keywords!
"""
        
        try:
            response = self.llm.invoke(planning_prompt)
            content = self._clean_json_response(response.content)
            result = json.loads(content)
            
            # Validate required fields
            if 'execution_steps' not in result:
                raise ValueError("Missing execution_steps in plan")

            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Execution planning error: {e}")
            return self._fallback_execution_plan(need_analysis)

    def _execute_and_integrate(self, original_query: str, execution_plan: dict, need_analysis: dict) -> str:
        """Step 3: Execute plan and intelligently integrate results"""
        
        execution_results = []
        
        # Execute each step according to plan
        for step in execution_plan['execution_steps']:
            try:
                print(f"🔧 Executing step {step['step_id']}: {step['tool_name']}")
                result = self._execute_single_step(step, original_query)
                
                execution_results.append({
                    'step_id': step['step_id'],
                    'tool_name': step['tool_name'],
                    'purpose': step['purpose'],
                    'result': result,
                    'status': 'success'
                })
                
                # Store result for future reference
                self.context.last_action_results[step['tool_name']] = result

            except Exception as e:
                    print(f"❌ Step {step['step_id']} failed: {e}")
                    execution_results.append({
                        'step_id': step['step_id'],
                        'tool_name': step['tool_name'],
                        'purpose': step['purpose'],
                        'result': f"Tool execution failed: {str(e)}",
                        'status': 'failed',
                        'error': str(e)
                    })
        
        # Intelligent result integration
        return self._intelligent_result_integration(
            original_query, execution_results, execution_plan, need_analysis
        )

    def _execute_single_step(self, step: dict, original_query: str) -> str:
        """Execute a single tool step"""
        tool_name = step['tool_name']
        tool_query = step['input_query']
        
        if tool_name == 'search_funding':
            return self._execute_funding_search_tool(tool_query)
        elif tool_name == 'web_search':
            return self._execute_web_search_tool(tool_query)
        elif tool_name == 'risk_assessment':
            return self._execute_risk_assessment_tool(tool_query)
        elif tool_name == 'market_analysis':
            return self._execute_market_analysis_tool(tool_query)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_funding_search_tool(self, query: str) -> str:
        """Execute funding search with company context"""
        if not self.context.has_complete_company_info():
            return """❌ To provide precise investment firm recommendations, I need more company information:

📋 **Required Information**:
- Company industry (e.g., AI, Biotech, Fintech)
- Target funding stage (e.g., Pre-Seed, Seed, Series A)
- Target market/country (e.g., UK, US, Germany)

Please provide this information and I'll recommend the most suitable investment firms."""

        try:
            from tools.funding_rag_tool import search_funding
            
            search_params = {
                'company_name': self.context.company_info.get('company_name', ''),
                'industry': self.context.company_info.get('industry', 'AI'),
                'country': self.context.company_info.get('country', 'UK'),
                'stage': self.context.company_info.get('stage', 'Pre-Seed'),
                'founded_year': self.context.company_info.get('founded_year', ''),
                'team_size': self.context.company_info.get('team_size', ''),
                'funding_rounds': self.context.company_info.get('funding_rounds', ''),
                'revenue': self.context.company_info.get('revenue', ''),
                'funding_amount': self.context.company_info.get('funding_amount', ''),
                'description': self.context.company_info.get('description', '')
            }
            
            result = search_funding(**search_params)
            return result
            
        except Exception as e:
            return f"Error searching investment firms: {str(e)}"

    def _execute_web_search_tool(self, query: str) -> str:
        """Execute web search tool"""
        try:
            from tools import WebSearchTool
            tool = WebSearchTool()
            result = tool.run(query)
            return result
        except Exception as e:
            return f"Error performing web search: {str(e)}"

    # def _execute_risk_assessment_tool1(self, query: str) -> str:
    #     """Execute risk assessment tool"""
    #     if not self.context.has_complete_company_info():
    #         return "❌ Please provide complete company information for accurate risk assessment."

    #     try:
    #         from tools.risk_predict_tool import check_eligibility
            
    #         # assessment_info = {
    #         #     'company_age': 2025 - self.context.company_info.get('founded_year', 2023),
    #         #     'annual_revenue': self.context.company_info.get('revenue', 0),
    #         #     'team_size': self.context.company_info.get('team_size', 5),
    #         #     'funding_rounds': self.context.company_info.get('funding_rounds', 0),
    #         #     'industry': self.context.company_info.get('industry', ''),
    #         #     'description': self.context.company_info.get('description', ''),
    #         #     'has_financial_records': True,
    #         #     'has_business_plan': True
    #         # }
    #         print("风险工具启用了...................................................")
    #         assessment_info_test = {
    #             'founded_at': self.context.company_info.get('founded_year', 2023),
    #             'category_list': self.context.company_info.get('industry', ''),
    #             'country_code': self.context.company_info.get('country', ''), # 这里要的是国家代码而不是国家名称，输入之后要进行解析
    #             'funding_total_usd': self.context.company_info.get('funding_amount', 0),# 这里输入的是欧元，但是建模中建模的是美元，需要转换为美元，并且因为输入中没有这个字段，等待新增中
    #             'funding_rounds': self.context.company_info.get('funding_rounds', 0)
    #         }
            
    #         result = check_eligibility(assessment_info_test)
    #         print('风险评估结果：',result)
    #         return result
            
    #     except Exception as e:
    #         print(f"Error during risk assessment: {str(e)}")
    #         return f"Error during risk assessment: {str(e)}"
    def _execute_risk_assessment_tool(self, query: str) -> str:
        """Execute risk assessment tool"""
        if not self.context.has_complete_company_info():
            return "❌ Please provide complete company information for accurate risk assessment."

        try:
            from tools.risk_predict_tool import check_eligibility
            
            print("risk_assessment_tool called...................................................")
            
            # 数据映射和转换
            def get_country_code(country_name):
                """将国家名称转换为国家代码"""
                country_mapping = {
                    'China': 'CHN', 'United States': 'USA', 'Germany': 'DEU',
                    'United Kingdom': 'GBR', 'France': 'FRA', 'Japan': 'JPN',
                    'Canada': 'CAN', 'Australia': 'AUS', 'India': 'IND',
                    'Brazil': 'BRA', 'Russia': 'RUS', 'South Korea': 'KOR',
                    'Italy': 'ITA', 'Spain': 'ESP', 'Netherlands': 'NLD',
                    'Switzerland': 'CHE', 'Sweden': 'SWE', 'Singapore': 'SGP',
                    'Israel': 'ISR', 'Norway': 'NOR'
                }
                return country_mapping.get(country_name, 'USA')  # 默认USA
            
            def convert_eur_to_usd(eur_amount):
                """欧元转美元（使用固定汇率）"""
                if eur_amount is None or eur_amount == 0:
                    return 0
                return float(eur_amount) * 1.08  # 近似汇率
            
            # 构造符合工具要求的数据格式
            assessment_info = {
                'founded_at': str(self.context.company_info.get('founded_year', 2023)),
                'category_list': self.context.company_info.get('industry', 'Technology'),
                'country_code': get_country_code(self.context.company_info.get('country', 'United States')),
                'funding_total_usd': convert_eur_to_usd(self.context.company_info.get('funding_amount', 0)),
                'funding_rounds': self.context.company_info.get('funding_rounds', 0)
            }
            
            print(f"assessment_info: {assessment_info}")
            
            result = check_eligibility(assessment_info)
            print('risk_assessment_tool result: ', result)
            return result
            
        except Exception as e:
            print(f"Error during risk assessment: {str(e)}")
            return f"❌ risk_assessment_tool error: {str(e)}"
    
    def _execute_market_analysis_tool(self, query: str) -> str:
        """Execute market analysis tool"""
        industry = self.context.company_info.get('industry', 'AI')
        country = self.context.company_info.get('country', 'UK')
        stage = self.context.company_info.get('stage', 'Pre-Seed')
        
        # Generate comprehensive market analysis
        market_analysis = f"""## 📊 {industry.upper()} INDUSTRY MARKET ANALYSIS

### 🔥 CURRENT MARKET TRENDS
- **Investment Activity**: {industry} sector shows {self._get_market_sentiment(industry, country)} activity in {country}
- **Funding Environment**: {stage} stage companies face {self._get_funding_difficulty(stage)} conditions
- **Competition Level**: {self._get_competition_level(industry)} competitive landscape

### 💰 FUNDING LANDSCAPE
- **Typical Range**: {self._get_typical_funding_range(stage)}
- **Investor Preferences**: Focus on {self._get_investor_preferences(industry)}
- **Success Factors**: {self._get_success_factors(industry, stage)}

### 📈 MARKET FORECAST
{self._generate_market_forecast(industry, country)}

### 💡 STRATEGIC RECOMMENDATIONS
{self._generate_personalized_market_advice(industry, stage, country)}
"""
        return market_analysis

    def _intelligent_result_integration(self, original_query: str, execution_results: list, 
                                      execution_plan: dict, need_analysis: dict) -> str:
        """Intelligently integrate multiple tool results"""
        
        integration_prompt = f"""
You are an expert information synthesizer. Create a comprehensive, valuable response.

【Original User Query】: "{original_query}"

【User Need Analysis】:
{json.dumps(need_analysis, indent=2, ensure_ascii=False)}

【Execution Results】:
{json.dumps(execution_results, indent=2, ensure_ascii=False)}

【Integration Strategy】:
{json.dumps(execution_plan.get('integration_strategy', {}), indent=2, ensure_ascii=False)}

Integration Requirements:

1. **Coherence**: Ensure smooth information flow
2. **Completeness**: Address the user's core question
3. **Accuracy**: Base on actual data, avoid speculation  
4. **Actionability**: Provide practical recommendations
5. **Professional**: Maintain investment consulting standards

Response Structure:
- 📋 **Executive Summary** (directly answer user's question)
- 📊 **Detailed Analysis** (based on tool results)
- 💡 **Key Insights & Recommendations** 
- 🔄 **Next Steps** (if applicable)

Quality Checks:
- Avoid redundant information
- Highlight the most important findings
- Ensure practical value for the user
- Maintain professional yet accessible tone

Generate the final integrated response:
"""
        
        try:
            response = self.llm.invoke(integration_prompt)
            return response.content
            
        except Exception as e:
            print(f"❌ Integration error: {e}")
            return self._fallback_integration(execution_results)

    # Utility Methods
    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract JSON"""
        content = content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        return content

    def _fallback_need_analysis(self, query: str) -> dict:
        """Fallback need analysis when LLM parsing fails"""
        return {
            "need_type": {"primary": "information_seeking", "secondary": []},
            "information_requirements": {
                "data_freshness": "recent",
                "data_type": "mixed",
                "detail_level": "moderate"
            },
            "mentioned_entities": {"companies": [], "industries": [], "locations": [], "stages": []},
            "user_context": {
                "satisfaction_with_previous": "no_previous",
                "follow_up_type": "new_topic",
                "references_previous_results": False
            },
            "urgency": "medium",
            "reasoning": "Fallback analysis due to parsing error"
        }

    def _fallback_execution_plan(self, need_analysis: dict) -> dict:
        """Fallback execution plan when planning fails"""
        # Simple rule-based fallback
        if any(word in str(need_analysis).lower() for word in ['investor', 'funding', 'recommend']):
            tool_name = 'search_funding'
        elif any(word in str(need_analysis).lower() for word in ['risk', 'assessment', 'ready']):
            tool_name = 'risk_assessment'
        elif any(word in str(need_analysis).lower() for word in ['market', 'trend', 'analysis']):
            tool_name = 'market_analysis'
        else:
            tool_name = 'web_search'
            
        return {
            "execution_strategy": {"approach": "single_tool", "rationale": "Fallback strategy"},
            "execution_steps": [{
                "step_id": 1,
                "tool_name": tool_name,
                "purpose": "Address user query",
                "input_query": str(need_analysis),
                "depends_on": None,
                "priority": "high"
            }],
            "integration_strategy": {"method": "direct_presentation", "focus_points": [], "format": "standard"},
            "expected_outcome": "Basic response to user query"
        }

    def _fallback_integration(self, execution_results: list) -> str:
        """Fallback integration when smart integration fails"""
        result_text = []
        for result in execution_results:
            if result['status'] == 'success':
                result_text.append(f"## {result['tool_name'].title()} Result\n{result['result']}")
        
        if not result_text:
            return "I apologize, but I encountered issues processing your request. Please try rephrasing your question."
        
        return "\n\n".join(result_text)

    # Market analysis helper methods (keeping existing logic)
    def _get_market_sentiment(self, industry: str, country: str) -> str:
        sentiment_map = {
            ('AI', 'UK'): "strong", ('AI', 'US'): "very active",
            ('Biotech', 'UK'): "stable", ('Fintech', 'UK'): "cautiously optimistic"
        }
        return sentiment_map.get((industry, country), "moderate")

    def _get_funding_difficulty(self, stage: str) -> str:
        difficulty_map = {
            'Pre-Seed': "manageable challenges with strong team required",
            'Seed': "moderate difficulty requiring product validation",
            'Series A': "higher difficulty requiring clear revenue"
        }
        return difficulty_map.get(stage, "standard preparation requirements")

    def _get_competition_level(self, industry: str) -> str:
        competition_map = {
            'AI': "intense", 'Biotech': "highly specialized",
            'Fintech': "extremely competitive", 'SaaS': "differentiation-critical"
        }
        return competition_map.get(industry, "standard")

    def _get_typical_funding_range(self, stage: str) -> str:
        range_map = {
            'Pre-Seed': "ranges from $500K-$2M",
            'Seed': "ranges from $1M-$5M", 
            'Series A': "ranges from $3M-$15M"
        }
        return range_map.get(stage, "varies by specific circumstances")

    def _get_investor_preferences(self, industry: str) -> str:
        preferences_map = {
            'AI': "technical innovation and practical application scenarios",
            'Biotech': "clinical validation and regulatory compliance",
            'Fintech': "regulatory compliance and user growth",
            'SaaS': "customer retention and ARR growth"
        }
        return preferences_map.get(industry, "business model validation and team strength")

    def _get_success_factors(self, industry: str, stage: str) -> str:
        if industry == 'AI' and stage == 'Pre-Seed':
            return "Technical prototype + Market demand validation + Experienced technical team"
        elif industry == 'Biotech':
            return "Scientific validation + IP protection + Clear regulatory pathway"
        else:
            return "Product-market fit + Scalable business model + Strong execution team"

    def _generate_market_forecast(self, industry: str, country: str) -> str:
        forecasts = {
            ('AI', 'UK'): "Investment activity expected to remain active, particularly in healthcare AI and enterprise AI sectors",
            ('Biotech', 'UK'): "Biotech investment expected to remain stable with government policy support driving growth",
            ('Fintech', 'UK'): "Regulatory clarity will create new investment opportunities, especially in digital banking and payments"
        }
        return forecasts.get((industry, country), f"{industry} sector in {country} expected to maintain stable investment environment")

    def _generate_personalized_market_advice(self, industry: str, stage: str, country: str) -> str:
        advice_parts = []
        
        if industry == 'AI':
            advice_parts.append("Emphasize your AI technology's uniqueness and practical application value")
        elif industry == 'Biotech':
            advice_parts.append("Highlight clinical data and regulatory progress")
        
        if stage == 'Pre-Seed':
            advice_parts.append("Focus on proving team capabilities and technical feasibility")
        elif stage == 'Seed':
            advice_parts.append("Demonstrate product-market fit and initial customer feedback")
        
        if country == 'UK':
            advice_parts.append("Leverage UK tax incentives and government innovation support")
        
        return "; ".join(advice_parts) + "."

    # Backward compatibility methods
    def get_conversation_summary(self) -> str:
        """Get conversation summary for backward compatibility"""
        if not self.context.conversation_history:
            return "No conversation history available"
        
        summary = f"""## 📋 CONVERSATION SUMMARY
        
**Session Duration**: {datetime.now() - self.context.session_start_time}
**Interaction Rounds**: {len(self.context.conversation_history)}
**Current State**: {self.context.current_state.value}

**Company Information Completeness**: {'✅ Complete' if self.context.has_complete_company_info() else '⚠️ Needs completion'}

**Analyses Performed**: {', '.join(self.context.last_action_results.keys()) if self.context.last_action_results else 'None yet'}

**Latest Focus**: {self.context.conversation_history[-1]['analysis_result'].get('need_type', {}).get('primary', 'None') if self.context.conversation_history else 'None'}
"""
        return summary

    def reset_conversation(self):
        """Reset conversation state for backward compatibility"""
        self.context = ConversationContext()
        self.memory.clear()
        print("🔄 Conversation state has been reset")

    # Legacy backward compatibility methods
    def _enhanced_intent_analysis(self, query: str) -> dict:
        """Legacy method - redirects to new need analysis"""
        need_analysis = self._analyze_user_need(query)
        
        # Convert new format to legacy format
        primary_intent = self._map_need_to_legacy_intent(need_analysis)
        
        return {
            'primary_intent': primary_intent,
            'secondary_intents': [],
            'confidence': 0.8,
            'reasoning': need_analysis.get('reasoning', 'Smart analysis'),
            'context_influence': 'Enhanced smart tool composition',
            'suggested_clarifications': [],
            'multi_tool_needed': len(need_analysis.get('mentioned_entities', {}).get('companies', [])) > 0,
            'need_analysis': need_analysis  # Store full analysis
        }

    def _map_need_to_legacy_intent(self, need_analysis: dict) -> str:
        """Map new need analysis to legacy intent types"""
        primary_need = need_analysis.get('need_type', {}).get('primary', 'information_seeking')
        entities = need_analysis.get('mentioned_entities', {})
        
        # If specific companies mentioned, likely want web search
        if entities.get('companies'):
            return 'follow_up_question'  # Will trigger web search in legacy handlers
        
        # Map based on primary need and context
        if primary_need == 'decision_support':
            if any(entities.get(key) for key in ['industries', 'stages']):
                return 'search_funding'
            return 'general_chat'
        elif primary_need == 'information_seeking':
            if entities.get('companies'):
                return 'follow_up_question'
            return 'general_chat'
        elif primary_need == 'problem_solving':
            return 'risk_assessment'
        else:
            return 'general_chat'

    def _execute_contextual_action(self, query: str, intent_result: dict) -> str:
        """Legacy method - enhanced with smart tool composition"""
        
        # If we have enhanced need analysis, use smart composition
        if 'need_analysis' in intent_result:
            need_analysis = intent_result['need_analysis']
            execution_plan = self._generate_execution_plan(need_analysis)
            return self._execute_and_integrate(query, execution_plan, need_analysis)
        
        # Fallback to legacy-style execution
        primary_intent = intent_result['primary_intent']
        
        if primary_intent == 'search_funding':
            return self._execute_legacy_funding_search(query, intent_result)
        elif primary_intent == 'risk_assessment':
            return self._execute_legacy_risk_assessment(query, intent_result)
        elif primary_intent == 'market_analysis':
            return self._execute_legacy_market_analysis(query, intent_result)
        elif primary_intent == 'follow_up_question':
            return self._execute_legacy_follow_up(query, intent_result)
        else:
            return self._execute_legacy_general_response(query, intent_result)

    def _execute_legacy_funding_search(self, query: str, intent_result: dict) -> str:
        """Legacy funding search execution"""
        result = self._execute_funding_search_tool(query)
        
        context_note = ""
        if intent_result.get('context_influence'):
            context_note = f"\n**Context Analysis**: {intent_result['context_influence']}\n"
        
        return f"""## 🎯 INTELLIGENT INVESTMENT FIRM RECOMMENDATIONS
        
**Analysis Confidence**: {intent_result.get('confidence', 0.8):.1%}
**Analysis Reasoning**: {intent_result.get('reasoning', 'Based on company information matching')}
{context_note}
{result}

## 💡 NEXT STEPS
If you have questions about the recommendations, you can:
- Ask "Why were these firms recommended?" for detailed explanations
- Say "Recommend different types of investors" for alternative suggestions
- Request "Assess my funding risk" for readiness analysis
"""

    def _execute_legacy_risk_assessment(self, query: str, intent_result: dict) -> str:
        """Legacy risk assessment execution"""
        result = self._execute_risk_assessment_tool(query)
        
        return f"""## 📊 INTELLIGENT RISK ASSESSMENT REPORT

**Analysis Confidence**: {intent_result.get('confidence', 0.8):.1%}
**Analysis Reasoning**: {intent_result.get('reasoning', 'Based on comprehensive company information assessment')}

{result}

## 🎯 STATE-BASED RECOMMENDATIONS
Based on current conversation state and assessment results:
- To find investors suitable for your risk profile, say "Recommend investors who accept our risk level"
- To understand improvement strategies, ask "How can I enhance my funding readiness"
- For industry benchmarking, say "Analyze risk levels of similar companies in my industry"
"""

    def _execute_legacy_market_analysis(self, query: str, intent_result: dict) -> str:
        """Legacy market analysis execution"""
        result = self._execute_market_analysis_tool(query)
        
        return f"""## 📊 INTELLIGENT MARKET ANALYSIS

**Analysis Confidence**: {intent_result.get('confidence', 0.8):.1%}
**Analysis Reasoning**: {intent_result.get('reasoning', 'Based on industry data and company situation')}

{result}
"""

    def _execute_legacy_follow_up(self, query: str, intent_result: dict) -> str:
        """Legacy follow-up execution - enhanced with smart web search"""
        
        # Check if user is asking about specific entities
        entities = self._extract_entities_from_query(query)
        
        if entities:
            # Use web search for specific entity information
            search_query = f"{entities[0]} company profile investment details"
            web_result = self._execute_web_search_tool(search_query)
            
            return f"""## 🔍 DETAILED INFORMATION

{web_result}

## 💡 RELATED SUGGESTIONS
Based on this information, you might also want to:
- Compare this with other similar firms
- Assess how well this matches your company profile
- Get recommendations for approaching this investor
"""
        else:
            # Handle general follow-up
            recent_context = self.context.get_recent_context(2)
            last_results = self.context.last_action_results
            
            follow_up_prompt = f"""
You are a professional funding advisor. The user has follow-up questions about previous responses.

User question: {query}
Recent conversation context: {recent_context}

Previous analysis results summary:
{json.dumps({k: str(v)[:300] + "..." for k, v in last_results.items()}, indent=2)}

Please provide professional, detailed answers to the user's follow-up questions.
"""
            
            try:
                response = self.llm.invoke(follow_up_prompt)
                return f"""## 💬 FOLLOW-UP RESPONSE

{response.content}

## 🔍 NEED MORE INFORMATION?
- For detailed analysis of specific investment firms, tell me which firms interest you
- To adjust recommendation criteria, specify your new requirements
- For other types of analysis, please ask directly
"""
            except Exception as e:
                return f"Error processing follow-up question: {str(e)}"

    def _execute_legacy_general_response(self, query: str, intent_result: dict) -> str:
        """Legacy general response execution"""
        try:
            contextual_prompt = f"""
You are a professional funding advisor AI assistant with smart contextual understanding.

User query: {query}
Current conversation state: {self.context.current_state.value}
Recent conversation context: {self.context.get_recent_context(2)}

Intent analysis results:
- Primary intent: {intent_result['primary_intent']}
- Confidence: {intent_result.get('confidence', 0.5):.1%}
- Analysis reasoning: {intent_result.get('reasoning', 'General consultation')}

Based on the context, provide professional, helpful funding-related advice.

Proactively suggest specific capabilities:
- For finding investment firms: "I can recommend the most suitable investment firms based on your situation"
- For risk assessment: "I can assess your company's current funding readiness"  
- For market analysis: "I can analyze the latest funding trends in your industry"
- For specific company research: "I can search for detailed information about any investment firm"
"""
            
            response = self.llm.invoke(contextual_prompt)
            
            follow_up_suggestions = self._generate_smart_suggestions()
            
            return f"""{response.content}

{follow_up_suggestions}"""
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entity names from query using pattern matching"""
        patterns = [
            r'(introduce|tell me about|what is|who is)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+for\s+me)?[.?]?$',
            r'(information about|details about|more about)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+for\s+me)?[.?]?$',
            r'([A-Z][a-zA-Z\s&]+(?:Campus|Ventures|Capital|Partners|Fund|Group|Accelerator))',
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    entities.extend([match[1].strip() if len(match) > 1 else match[0].strip() for match in matches])
                else:
                    entities.extend([match.strip() for match in matches])
        
        return list(set(entities))

    def _generate_smart_suggestions(self) -> str:
        """Generate smart follow-up suggestions"""
        suggestions_map = {
            ConversationState.INITIAL: """
## 🚀 START YOUR FUNDING JOURNEY
I can help you with:
- 📋 **Company Information** - Tell me your industry, stage, target market
- 🎯 **Investment Firm Recommendations** - Find the most suitable investors
- 📊 **Funding Risk Assessment** - Analyze your readiness and success probability
- 🌐 **Research Specific Firms** - Get detailed information about any investment firm
""",
            ConversationState.COMPANY_INFO_GATHERING: """
## 📋 COMPLETE COMPANY INFORMATION
To provide more precise recommendations, please provide:
- Company industry sector
- Target funding stage  
- Primary market region
""",
            ConversationState.FUNDING_SEARCH: """
## 🎯 EXPLORE RECOMMENDATION RESULTS
You can:
- Ask for detailed information about specific investment firms
- Request different types of investor recommendations
- Learn how to approach these investment firms
- Get risk assessment for your funding prospects
""",
            ConversationState.FOLLOW_UP: """
## 💬 CONTINUE DEEP DISCUSSION
You can:
- Research specific companies mentioned in results
- Ask any specific questions about funding
- Request detailed explanations of previous analysis
- Explore other funding strategy options
"""
        }
        
        return suggestions_map.get(self.context.current_state, """
## 💡 HOW I CAN HELP
- 🔍 **Research Companies**: "Tell me about [Company Name]"
- 🎯 **Find Investors**: "Recommend investment firms for my startup"
- 📊 **Assess Risk**: "Evaluate my funding readiness"
- 📈 **Market Analysis**: "Analyze trends in my industry"
""")

    # Additional legacy compatibility methods
    def _calculate_keyword_scores(self, query: str) -> dict:
        """Legacy keyword scoring - now enhanced"""
        # Keep for backward compatibility but not actively used
        return {"legacy_method": "replaced_by_smart_analysis"}

    def _analyze_context_clues(self, query: str, recent_context: str) -> dict:
        """Legacy context clues - now enhanced"""
        # Keep for backward compatibility but not actively used
        return {"legacy_method": "replaced_by_smart_analysis"}

    def _fallback_intent_analysis(self, query: str, keyword_scores: dict) -> dict:
        """Legacy fallback method"""
        return {
            'primary_intent': 'general_chat',
            'secondary_intents': [],
            'confidence': 0.5,
            'reasoning': 'Fallback analysis',
            'context_influence': 'Limited analysis available',
            'suggested_clarifications': [],
            'multi_tool_needed': False
        }


# Create alias for backward compatibility
FundingAgent = SmartFundingAgent

