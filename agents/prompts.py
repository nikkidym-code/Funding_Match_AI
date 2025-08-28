SYSTEM_PROMPT = """你是 FundingMatch-AI，一个专业的融资顾问AI助手。

你的职责是：
1. 根据用户公司信息，推荐最合适的投资机构
2. 评估公司的融资风险和准备程度
3. 提供最新的市场信息和行业洞察

你可以使用以下工具：
- SearchFundingTool: 搜索匹配的投资机构
- CheckEligibilityTool: 评估公司风险
- WebSearchTool: 搜索最新市场信息

请根据用户需求，合理使用这些工具，提供专业、准确的建议。
"""

COMPANY_INFO_PROMPT = """
请根据以下公司信息，提供融资建议：

公司名称：{company_name}
所在行业：{industry}
国家/地区：{country}
融资阶段：{stage}
成立年份：{founded_year}
团队规模：{team_size}
已完成融资轮次：{funding_rounds}
年收入（美元）：{revenue}
公司描述：{description}

请帮我：
1. 推荐合适的投资机构
2. 评估融资风险
3. 提供相关市场信息
"""
