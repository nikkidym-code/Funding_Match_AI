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
        
        # country code mapping (main countries)
        self.country_mapping = {
            'China': 'CHN', 'United States': 'USA', 'Germany': 'DEU',
            'United Kingdom': 'GBR', 'France': 'FRA', 'Japan': 'JPN',
            'Canada': 'CAN', 'Australia': 'AUS', 'India': 'IND',
            'Brazil': 'BRA', 'Russia': 'RUS', 'South Korea': 'KOR',
            'Italy': 'ITA', 'Spain': 'ESP', 'Netherlands': 'NLD',
            'Switzerland': 'CHE', 'Sweden': 'SWE', 'Singapore': 'SGP',
            'Israel': 'ISR', 'Norway': 'NOR'
        }
        
        # industry category mapping
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
        
        # basic feature statistics (for missing value filling and normalization)
        self.feature_stats = {
            'founded_year_median': 2010,
            'funding_total_usd_median': 2000000,
            'funding_rounds_median': 2,
            'top_countries': ['USA', 'GBR', 'CAN', 'DEU', 'FRA', 'IND', 'CHN', 'ISR', 'AUS', 'NLD'],
            'eur_to_usd_rate': 1.08  # eur to usd rate (approximate value)
        }
        
    def load_model(self):
        """load trained model and feature names"""
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
        """preprocess company data"""
        processed_data = {}
        
        # 1. process founded year
        founded_year = company_info.get('founded_at', 2023)
        if isinstance(founded_year, str):
            try:
                founded_year = pd.to_datetime(founded_year).year
            except:
                founded_year = int(founded_year) if founded_year.isdigit() else 2023
        processed_data['founded_year'] = founded_year
        
        # 2. process funding amount (eur to usd)
        funding_eur = company_info.get('funding_total_usd', 0)
        if isinstance(funding_eur, str):
            funding_eur = float(funding_eur.replace(',', '')) if funding_eur.replace(',', '').replace('.', '').isdigit() else 0
        funding_usd = funding_eur * self.feature_stats['eur_to_usd_rate']
        processed_data['funding_total_usd'] = funding_usd
        
        # 3. process funding rounds
        processed_data['funding_rounds'] = company_info.get('funding_rounds', 0)
        
        # 4. process country code
        country = company_info.get('country_code', 'Unknown')
        if country in self.country_mapping.values():
            country_code = country
        elif country in self.country_mapping.keys():
            country_code = self.country_mapping[country]
        else:
            country_code = 'Other'
        
        # check if in top countries
        if country_code in self.feature_stats['top_countries']:
            processed_data['country_grouped'] = country_code
        else:
            processed_data['country_grouped'] = 'Other'
        
        # 5. process industry category
        category = company_info.get('category_list', 'Technology')
        if category in self.industry_mapping:
            processed_data['main_category'] = self.industry_mapping[category]
        else:
            processed_data['main_category'] = category
        
        return processed_data
    
    def create_feature_vector(self, processed_data):
        """create feature vector"""
        if not self.is_loaded:
            return None
            
            # create basic numerical features
        feature_vector = {}
        
        # numerical features
        feature_vector['founded_year'] = processed_data.get('founded_year', self.feature_stats['founded_year_median'])
        feature_vector['funding_total_usd'] = processed_data.get('funding_total_usd', self.feature_stats['funding_total_usd_median'])
        feature_vector['funding_rounds'] = processed_data.get('funding_rounds', self.feature_stats['funding_rounds_median'])
        
        # initialize all features to 0
        for feature_name in self.feature_names:
            if feature_name not in feature_vector:
                feature_vector[feature_name] = 0
        
        # set corresponding categorical features to 1
        country_feature = f"country_{processed_data.get('country_grouped', 'Other')}"
        if country_feature in feature_vector:
            feature_vector[country_feature] = 1
            
        category_feature = f"category_{processed_data.get('main_category', 'Software')}"
        if category_feature in feature_vector:
            feature_vector[category_feature] = 1
        
        # convert to DataFrame
        df = pd.DataFrame([feature_vector])
        return df[self.feature_names]  # ensure feature order is correct
    
    def rule_based_assessment(self, processed_data):
        """rule-based risk assessment (when model is not available)"""
        risk_score = 0.0
        risk_factors = []
        
        # company age factor
        company_age = 2025 - processed_data.get('founded_year', 2023)
        if company_age < 2:
            risk_score += 0.3
            risk_factors.append("company founded time is short (high risk)")
        elif company_age > 10:
            risk_score -= 0.1
            risk_factors.append("company founded time is long (low risk)")
        
        # funding situation
        funding = processed_data.get('funding_total_usd', 0)
        if funding < 100000:  # less than 100,000 usd
            risk_score += 0.25
            risk_factors.append("funding amount is low (high risk)")
        elif funding > 5000000:  # more than 5,000,000 usd
            risk_score -= 0.15
            risk_factors.append("funding amount is enough (low risk)")
        
        # funding rounds
        rounds = processed_data.get('funding_rounds', 0)
        if rounds == 0:
            risk_score += 0.2
            risk_factors.append("no funding history (high risk)")
        elif rounds >= 3:
            risk_score -= 0.1
            risk_factors.append("multiple funding rounds (low risk)")
        
        # country factor
        country = processed_data.get('country_grouped', 'Other')
        if country == 'Other':
            risk_score += 0.1
            risk_factors.append("non-major startup market (slight high risk)")
        elif country in ['USA', 'CHN', 'GBR']:
            risk_score -= 0.05
            risk_factors.append("major startup market (slight low risk)")
        
        # limit risk score between 0-1
        risk_score = max(0.0, min(1.0, 0.5 + risk_score))
        
        return risk_score, risk_factors
    
    def predict_risk(self, company_info):
        """predict company risk"""
        # preprocess data
        processed_data = self.preprocess_company_data(company_info)
        
        if self.is_loaded and self.model is not None:
            try:
                    # use machine learning model
                feature_vector = self.create_feature_vector(processed_data)
                risk_probability = self.model.predict_proba(feature_vector)[0][1]
                
                # get feature importance explanation (simplified version)
                explanations = self._get_model_explanations(processed_data, risk_probability)
                
                return {
                    'risk_probability': float(risk_probability),
                    'risk_level': 'HIGH' if risk_probability > 0.6 else 'MEDIUM' if risk_probability > 0.4 else 'LOW',
                    'explanations': explanations,
                    'method': 'ML_MODEL'
                }
            except Exception as e:
                print(f"model prediction failed, using rule-based method: {str(e)}")
        
        # use rule-based method
        risk_score, risk_factors = self.rule_based_assessment(processed_data)
        
        return {
            'risk_probability': risk_score,
            'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.4 else 'LOW',
            'explanations': risk_factors,
            'method': 'RULE_BASED'
        }
    
    def _get_model_explanations(self, processed_data, risk_prob):
        """get model prediction explanation"""
        explanations = []
        
        # give explanation based on data features
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
                explanations.append("• sufficient funding,有利于业务扩展")  # for business expansion
            if rounds >= 3:
                explanations.append("• multiple funding rounds, high investor recognition")
        
        return explanations

# create global instance
_risk_tool = None

def get_risk_tool():
    """get risk assessment tool instance"""
    global _risk_tool
    if _risk_tool is None:
        _risk_tool = RiskAssessmentTool()
        _risk_tool.load_model()
    return _risk_tool

def check_eligibility(company_info):
    """main external interface function"""
    try:
        tool = get_risk_tool()
        result = tool.predict_risk(company_info)
        
        # format output
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
        
        # add advice
        if result['risk_level'] == 'HIGH':
            output += "\n💡 advice: focus on funding flow, business model validation, and team stability"
        elif result['risk_level'] == 'MEDIUM':
            output += "\n💡 advice: closely monitor key performance indicators, develop risk mitigation plans"
        else:
            output += "\n💡 advice: stay on the current development track, consider further expansion"
        
        return output.strip()
        
    except Exception as e:
        return f"❌ risk assessment error: {str(e)}"

# tool definition
class EligibilityQuery(BaseModel):
    company_info: Dict = Field(description="Company information matching your input form fields")


# create tool
CheckEligibilityTool = StructuredTool.from_function(
    name="CheckEligibilityTool",
    description="Startup risk assessment based on real startup data model with enhanced field analysis",
    func=check_eligibility,
    args_schema=EligibilityQuery
)