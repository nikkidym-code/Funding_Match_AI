# tools/risk_predict_tool.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from typing import Dict
import warnings

warnings.filterwarnings('ignore')

class RiskAssessmentTool:
    def __init__(self, model_path="risk_model.joblib", features_path="model_feature_names.joblib"):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_names = None
        self.is_loaded = False
        
        # 国家代码映射（主要国家）
        self.country_mapping = {
            'China': 'CHN', 'United States': 'USA', 'Germany': 'DEU',
            'United Kingdom': 'GBR', 'France': 'FRA', 'Japan': 'JPN',
            'Canada': 'CAN', 'Australia': 'AUS', 'India': 'IND',
            'Brazil': 'BRA', 'Russia': 'RUS', 'South Korea': 'KOR',
            'Italy': 'ITA', 'Spain': 'ESP', 'Netherlands': 'NLD',
            'Switzerland': 'CHE', 'Sweden': 'SWE', 'Singapore': 'SGP',
            'Israel': 'ISR', 'Norway': 'NOR'
        }
        
        # 行业类别映射
        self.industry_mapping = {
            'Technology': 'Software',
            'Healthcare': 'Biotechnology',
            'Finance': 'Financial Services',
            'Education': 'E-Learning',
            'Retail': 'E-Commerce',
            'Manufacturing': 'Hardware + Software',
            'Energy': 'CleanTech',
            'Transportation': 'Automotive',
            'Media': 'Advertising',
            'Real Estate': 'Real Estate',
            'Food': 'Food and Beverages'
        }
        
        # 基础特征统计（用于缺失值填充和正常化）
        self.feature_stats = {
            'founded_year_median': 2010,
            'funding_total_usd_median': 2000000,
            'funding_rounds_median': 2,
            'top_countries': ['USA', 'GBR', 'CAN', 'DEU', 'FRA', 'IND', 'CHN', 'ISR', 'AUS', 'NLD'],
            'eur_to_usd_rate': 1.08  # 欧元到美元汇率（近似值）
        }
        
    def load_model(self):
        """加载训练好的模型和特征名称"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.features_path):
                self.model = joblib.load(self.model_path)
                self.feature_names = joblib.load(self.features_path)
                self.is_loaded = True
                print("✅ risk_predict_tool model loaded successfully")
                return True
            else:
                print("⚠️ risk_predict_tool model file not found, using simplified rules")
                return False
        except Exception as e:
            print(f"❌ risk_predict_tool model loading failed: {str(e)}")
            return False
    
    def preprocess_company_data(self, company_info):
        """预处理公司数据"""
        processed_data = {}
        
        # 1. 处理成立年份
        founded_year = company_info.get('founded_at', 2023)
        if isinstance(founded_year, str):
            try:
                founded_year = pd.to_datetime(founded_year).year
            except:
                founded_year = int(founded_year) if founded_year.isdigit() else 2023
        processed_data['founded_year'] = founded_year
        
        # 2. 处理融资金额（欧元转美元）
        funding_eur = company_info.get('funding_total_usd', 0)
        if isinstance(funding_eur, str):
            funding_eur = float(funding_eur.replace(',', '')) if funding_eur.replace(',', '').replace('.', '').isdigit() else 0
        funding_usd = funding_eur * self.feature_stats['eur_to_usd_rate']
        processed_data['funding_total_usd'] = funding_usd
        
        # 3. 处理融资轮数
        processed_data['funding_rounds'] = company_info.get('funding_rounds', 0)
        
        # 4. 处理国家代码
        country = company_info.get('country_code', 'Unknown')
        if country in self.country_mapping.values():
            country_code = country
        elif country in self.country_mapping.keys():
            country_code = self.country_mapping[country]
        else:
            country_code = 'Other'
        
        # 检查是否在top国家中
        if country_code in self.feature_stats['top_countries']:
            processed_data['country_grouped'] = country_code
        else:
            processed_data['country_grouped'] = 'Other'
        
        # 5. 处理行业类别
        category = company_info.get('category_list', 'Technology')
        if category in self.industry_mapping:
            processed_data['main_category'] = self.industry_mapping[category]
        else:
            processed_data['main_category'] = category
        
        return processed_data
    
    def create_feature_vector(self, processed_data):
        """创建特征向量"""
        if not self.is_loaded:
            return None
            
        # 创建基础数值特征
        feature_vector = {}
        
        # 数值特征
        feature_vector['founded_year'] = processed_data.get('founded_year', self.feature_stats['founded_year_median'])
        feature_vector['funding_total_usd'] = processed_data.get('funding_total_usd', self.feature_stats['funding_total_usd_median'])
        feature_vector['funding_rounds'] = processed_data.get('funding_rounds', self.feature_stats['funding_rounds_median'])
        
        # 初始化所有特征为0
        for feature_name in self.feature_names:
            if feature_name not in feature_vector:
                feature_vector[feature_name] = 0
        
        # 设置对应的分类特征为1
        country_feature = f"country_{processed_data.get('country_grouped', 'Other')}"
        if country_feature in feature_vector:
            feature_vector[country_feature] = 1
            
        category_feature = f"category_{processed_data.get('main_category', 'Software')}"
        if category_feature in feature_vector:
            feature_vector[category_feature] = 1
        
        # 转换为DataFrame
        df = pd.DataFrame([feature_vector])
        return df[self.feature_names]  # 确保特征顺序正确
    
    def rule_based_assessment(self, processed_data):
        """基于规则的风险评估（当模型不可用时）"""
        risk_score = 0.0
        risk_factors = []
        
        # 公司年龄因素
        company_age = 2025 - processed_data.get('founded_year', 2023)
        if company_age < 2:
            risk_score += 0.3
            risk_factors.append("company founded time is short (high risk)")
        elif company_age > 10:
            risk_score -= 0.1
            risk_factors.append("company founded time is long (low risk)")
        
        # 融资情况
        funding = processed_data.get('funding_total_usd', 0)
        if funding < 100000:  # 10万美元以下
            risk_score += 0.25
            risk_factors.append("funding amount is low (high risk)")
        elif funding > 5000000:  # 500万美元以上
            risk_score -= 0.15
            risk_factors.append("funding amount is enough (low risk)")
        
        # 融资轮数
        rounds = processed_data.get('funding_rounds', 0)
        if rounds == 0:
            risk_score += 0.2
            risk_factors.append("no funding history (high risk)")
        elif rounds >= 3:
            risk_score -= 0.1
            risk_factors.append("multiple funding rounds (low risk)")
        
        # 国家因素
        country = processed_data.get('country_grouped', 'Other')
        if country == 'Other':
            risk_score += 0.1
            risk_factors.append("non-major startup market (slight high risk)")
        elif country in ['USA', 'CHN', 'GBR']:
            risk_score -= 0.05
            risk_factors.append("major startup market (slight low risk)")
        
        # 限制风险分数在0-1之间
        risk_score = max(0.0, min(1.0, 0.5 + risk_score))
        
        return risk_score, risk_factors
    
    def predict_risk(self, company_info):
        """预测公司风险"""
        # 预处理数据
        processed_data = self.preprocess_company_data(company_info)
        
        if self.is_loaded and self.model is not None:
            try:
                # 使用机器学习模型
                feature_vector = self.create_feature_vector(processed_data)
                risk_probability = self.model.predict_proba(feature_vector)[0][1]
                
                # 获取特征重要性解释（简化版）
                explanations = self._get_model_explanations(processed_data, risk_probability)
                
                return {
                    'risk_probability': float(risk_probability),
                    'risk_level': 'HIGH' if risk_probability > 0.6 else 'MEDIUM' if risk_probability > 0.4 else 'LOW',
                    'explanations': explanations,
                    'method': 'ML_MODEL'
                }
            except Exception as e:
                print(f"model prediction failed, using rule-based method: {str(e)}")
        
        # 使用基于规则的方法
        risk_score, risk_factors = self.rule_based_assessment(processed_data)
        
        return {
            'risk_probability': risk_score,
            'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.4 else 'LOW',
            'explanations': risk_factors,
            'method': 'RULE_BASED'
        }
    
    def _get_model_explanations(self, processed_data, risk_prob):
        """获取模型预测解释"""
        explanations = []
        
        # 基于数据特征给出解释
        company_age = 2025 - processed_data.get('founded_year', 2023)
        funding = processed_data.get('funding_total_usd', 0)
        rounds = processed_data.get('funding_rounds', 0)
        
        if risk_prob > 0.6:
            explanations.append("🚨 HIGH RISK - model predicts this company has a high probability of failure")
            if company_age < 3:
                explanations.append("• company founded time is short, lack of market validation")
            if funding < 500000:
                explanations.append("• funding amount is not enough, may face funding chain risk")
            if rounds <= 1:
                explanations.append("• fewer funding rounds, limited investor confidence")
        elif risk_prob > 0.4:
            explanations.append("⚠️ MEDIUM RISK - model predicts this company has a moderate risk")
            explanations.append("• the company has some potential, but needs to pay attention to key indicators")
        else:
            explanations.append("✅ LOW RISK - model predicts this company has a low risk")
            if company_age >= 5:
                explanations.append("• the company is relatively mature, with a stable market position")
            if funding > 2000000:
                explanations.append("• sufficient funding,有利于业务扩展")
            if rounds >= 3:
                explanations.append("• multiple funding rounds, high investor recognition")
        
        return explanations

# 创建全局实例
_risk_tool = None

def get_risk_tool():
    """获取风险评估工具实例"""
    global _risk_tool
    if _risk_tool is None:
        _risk_tool = RiskAssessmentTool()
        _risk_tool.load_model()
    return _risk_tool

def check_eligibility(company_info):
    """主要的对外接口函数"""
    try:
        tool = get_risk_tool()
        result = tool.predict_risk(company_info)
        
        # 格式化输出
        risk_emoji = "🚨" if result['risk_level'] == 'HIGH' else "⚠️" if result['risk_level'] == 'MEDIUM' else "✅"
        
        output = f"""
{risk_emoji} startup risk assessment result

📊 risk probability: {result['risk_probability']:.3f} ({result['risk_probability']*100:.1f}%)
🎯 risk level: {result['risk_level']}
🔍 evaluation method: {'machine learning model' if result['method'] == 'ML_MODEL' else 'rule engine'}

📝 detailed explanation:
"""
        
        for explanation in result['explanations']:
            output += f"{explanation}\n"
        
        # 添加建议
        if result['risk_level'] == 'HIGH':
            output += "\n💡 advice: focus on funding flow, business model validation, and team stability"
        elif result['risk_level'] == 'MEDIUM':
            output += "\n💡 advice: closely monitor key performance indicators, develop risk mitigation plans"
        else:
            output += "\n💡 advice: stay on the current development track, consider further expansion"
        
        return output.strip()
        
    except Exception as e:
        return f"❌ risk assessment error: {str(e)}"

# 工具定义
class EligibilityQuery(BaseModel):
    company_info: Dict = Field(description="Company information matching your input form fields")


# 创建工具
CheckEligibilityTool = StructuredTool.from_function(
    name="CheckEligibilityTool",
    description="Startup risk assessment based on real startup data model with enhanced field analysis",
    func=check_eligibility,
    args_schema=EligibilityQuery
)