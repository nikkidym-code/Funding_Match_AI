# funding_rag_tool.py
import os
import pandas as pd
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# Initialize OpenAI and ChromaDB
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chromadb = PersistentClient(path="./chroma_db_v2")
collection = chromadb.get_or_create_collection("funding_rag")

# Embedding generation function
def get_embedding(text: str, model="text-embedding-3-small", dimensions=512):
    response = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions
    )
    return response.data[0].embedding

# Build vector database with enhanced English text
def build_vector_store(csv_paths: List[str]):
    if collection.count() > 0:
        print("Vector database already exists, skipping construction")
        return
    
    successful_imports = 0
    failed_imports = 0
    
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"File does not exist: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        print(f"Processing file: {csv_path} with {len(df)} records")
        
        if 'Investment Areas' in df.columns and 'Funding Stage' in df.columns:
            df = df.dropna(subset=["Investment Areas", "Funding Stage"])
            
            for i, row in df.iterrows():
                try:
                    doc_id = str(row.get("rowID", f"{csv_path}_{i}"))
                    
                    # Build enhanced English text for vectorization
                    text_parts = []
                    
                    # Basic information
                    text_parts.append(f"Investment firm name: {row['Name']}")
                    text_parts.append(f"Located in: {row['Country']}")
                    
                    # Investor type
                    if pd.notna(row.get('Investor Type')) and row.get('Investor Type') != 'N/A':
                        text_parts.append(f"Investor type: {row['Investor Type']}")
                    
                    # Investment areas
                    text_parts.append(f"Investment focus areas: {row['Investment Areas']}")
                    
                    # Funding stage
                    text_parts.append(f"Funding stages: {row['Funding Stage']}")
                    
                    # Typical investment amount
                    if pd.notna(row.get('Typical Investment')) and row.get('Typical Investment') != 'N/A':
                        text_parts.append(f"Typical investment amount: {row['Typical Investment']}")
                    
                    # Combine full text
                    full_text = ". ".join(text_parts) + "."
                    
                    embedding = get_embedding(full_text)
                    
                    collection.add(
                        documents=[full_text],
                        ids=[doc_id],
                        embeddings=[embedding],
                        metadatas=[{
                            "name": str(row["Name"]),
                            "link": str(row.get("Profile Link", "")),
                            "stage": str(row["Funding Stage"]),
                            "areas": str(row["Investment Areas"]),
                            "country": str(row["Country"]),
                            "investor_type": str(row.get("Investor Type", "N/A")),
                            "funding_amount": str(row.get("Typical Funding Amount", "N/A"))
                        }]
                    )
                    
                    successful_imports += 1
                    
                    if successful_imports % 100 == 0:
                        print(f"Processed {successful_imports} records...")
                        
                except Exception as e:
                    failed_imports += 1
                    print(f"Error processing record {i}: {e}")
                    continue
    
    print(f"✅ Vector database construction completed!")
    print(f"  Successfully imported: {successful_imports} records")
    print(f"  Failed: {failed_imports} records") 
    print(f"  Total database entries: {collection.count()}")

# Parameter model - English version
class FundingQuery(BaseModel):
    industry: str = Field(description="Company industry")
    country: str = Field(description="Target country") 
    stage: str = Field(description="Funding stage")
    description: str = Field(default="", description="Detailed company description")

# Enhanced query function with English output
def search_funding(company_name: str, industry: str, country: str, stage: str,founded_year: str,team_size: str,funding_rounds: str,revenue: str,funding_amount: str, description: str = "") -> str:
    """Search for matching funding institutions"""
    
    print("=" * 50)
    print("SearchFundingTool called")
    print(f"Parameter type check:")
    print(f"company_name: {type(company_name)} = '{company_name}'")
    print(f"founded_year: {type(founded_year)} = '{founded_year}'")
    print(f"team_size: {type(team_size)} = '{team_size}'")
    print(f"funding_rounds: {type(funding_rounds)} = '{funding_rounds}'")
    print(f"revenue: {type(revenue)} = '{revenue}'")
    print(f"funding_amount: {type(funding_amount)} = '{funding_amount}'")
    print(f"industry: {type(industry)} = '{industry}'")
    print(f"country: {type(country)} = '{country}'")
    print(f"stage: {type(stage)} = '{stage}'")
    print(f"description: {type(description)} = '{description}'")
    print("=" * 50)
    
    # Check if parameters are incorrectly serialized
    if any(isinstance(param, str) and param.startswith('{') for param in [industry, country, stage, description,founded_year,team_size,funding_rounds,revenue,funding_amount]):
        return "Error: Parameters incorrectly serialized, please check Agent configuration"
    
    try:
        # Build enhanced query text
        query_parts = []
        
        if description:
            query_parts.append(f"Company description: {description}")
        
        query_parts.append(f"Industry: {industry}")
        query_parts.append(f"Target market: {country}")
        query_parts.append(f"Funding stage: {stage}")
        query_parts.append(f"Founded year: {founded_year}")
        query_parts.append(f"Team size: {team_size}")
        query_parts.append(f"Funding rounds: {funding_rounds}")
        query_parts.append(f"Revenue: {revenue}")
        query_parts.append(f"Funding amount: {funding_amount}")
        
        query_text = ". ".join(query_parts) + "."
        
        # If ChromaDB is empty, return mock data
        if collection.count() == 0:
            print("Warning: Vector database is empty, returning mock data")
            return generate_mock_results(industry, country, stage, description,founded_year,team_size,funding_rounds,revenue,funding_amount)
        
        query_embedding = get_embedding(query_text)
        
        # Query vector database
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(20, collection.count()),  # Get more results for filtering
            include=["metadatas", "distances", "documents"]
        )
        # print("resultsssssssss",results)
        funds = []
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for meta, dist, doc in zip(results['metadatas'][0], results['distances'][0], results['documents'][0]):
                score = 1 - dist
                
                # Apply filtering based on criteria
                country_match = country.lower() in meta.get('country', '').lower() if country != "Any" else True
                stage_match = any(s.strip().lower() in meta.get('stage', '').lower() 
                                for s in stage.replace(',', ' ').split()) if stage != "Any" else True
                # print("meta",meta.get('typical_funding_amount', 'N/A'),)
                funds.append({
                    "name": meta['name'],
                    "link": meta['link'],
                    "stage": meta['stage'],
                    "areas": meta['areas'],
                    "country": meta.get('country', 'Unknown'),
                    "investor_type": meta.get('investor_type', 'N/A'),
                    "funding_amount": meta.get('typical_funding_amount', 'N/A'),
                    "score": round(score, 3),
                    "description": doc,
                    "country_match": country_match,
                    "stage_match": stage_match
                })
        
        # Sort by relevance and apply filters
        filtered_funds = [f for f in funds if f['country_match'] or f['stage_match']]
        if not filtered_funds:
            filtered_funds = funds  # Fall back to all results if no matches
            
        top_funds = sorted(filtered_funds, key=lambda x: -x['score'])[:5]
        
        if not top_funds:
            return generate_mock_results(industry, country, stage, description,founded_year,team_size,funding_rounds,revenue,funding_amount)
        
        # Generate AI-enhanced recommendation using GPT-4
        context = ""
        for i, fund in enumerate(top_funds, 1):
            # print("fund",fund)
            context += (
                f"Investment Firm #{i}:\n"
                f"Name: {fund['name']}\n"
                f"Country: {fund['country']}\n"
                f"Type: {fund['investor_type']}\n"
                f"Investment Areas: {fund['areas']}\n"
                f"Funding Stages: {fund['stage']}\n"
                f"Typical Investment: {fund['funding_amount']}\n"
                f"Relevance Score: {fund['score']}\n"
                f"Profile Link: {fund['link']}\n\n"
            )
        
        # Use GPT-4 to generate sophisticated recommendations
        prompt = f"""You are a senior venture capital advisor with deep expertise in startup-investor matching. Analyze the following company and investment firms data to provide highly personalized, intelligent recommendations.

## COMPANY PROFILE
- **Company name:** {company_name}
- **Industry:** {industry}
- **Target Market:** {country}  
- **Funding Stage:** {stage}
- **Description:** {description}
- **Founded year:** {founded_year}
- **Team size:** {team_size}
- **Funding rounds:** {funding_rounds}
- **Revenue:** {revenue}
- **Funding amount(euro):** {funding_amount}

## AVAILABLE INVESTMENT FIRMS
{context}

## INSTRUCTIONS FOR INTELLIGENT ANALYSIS

Please provide a comprehensive report with the following structure:

### 🎯 EXECUTIVE SUMMARY
Write 2-3 sentences analyzing the overall funding landscape for this specific company, highlighting the most promising opportunities and any market dynamics.

### 📊 TOP INVESTMENT FIRM RECOMMENDATIONS

For the 3 most suitable firms, provide this exact structure:

**[Firm Number]. [Firm Name]**

**Firm Profile:**
- Location: [Country]
- Investment Focus: [Investment Areas] 
- Funding Stages: [Stages they invest in]
- Typical Investment: [Investment amount]
- Investor Type: [Type of investor]

**Why This Firm Matches:** [Write a complete paragraph explaining specifically why this firm is a good match for this company. Reference the company's industry, stage, description details, and how they align with the firm's investment criteria. Be specific about the connections - don't use generic language.]

**Strategic Approach:** [Provide 2-3 specific, actionable recommendations for how to approach this firm, considering their investment style, portfolio, and preferences.]

### 💡 KEY STRATEGIC INSIGHTS
Instead of generic advice, analyze this specific situation:
- What makes this company attractive to these particular investors?
- What potential concerns might these investors have?
- How should the company position itself given the current market conditions in their industry and geography?
- What alternative funding strategies should be considered if these don't work out?

### ⚠️ MARKET CONSIDERATIONS  
Provide specific insights about:
- Current investment climate for this industry in this geography
- Timeline expectations for this funding stage
- Any sector-specific trends that could impact fundraising

Be highly specific, avoid generic advice, and make every recommendation tailored to this exact company-investor combination. Reference specific details from both the company description and firm profiles."""

        try:
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500
            )
            
            ai_recommendation = gpt_response.choices[0].message.content.strip()
            
            # Add firm links section at the end
            links_section = "\n\n---\n\n### 🔗 INVESTMENT FIRM PROFILES\n\n"
            for i, fund in enumerate(top_funds[:3], 1):
                firm_link = fund['link'] if fund['link'] and fund['link'] != 'nan' else "Contact information available upon request"
                links_section += f"**{fund['name']}**\n"
                links_section += f"📍 {fund['country']} | 💰 {fund['funding_amount']} | 🏷️ {fund['investor_type']}\n"
                links_section += f"🌐 {firm_link}\n\n"
            
            # Combine AI recommendation with structured data and links
            output = f"# 📈 INVESTMENT STRATEGY REPORT\n"
            output += f"**Company Profile:** {industry} | {country} | {stage}\n\n"
            output += ai_recommendation
            output += links_section
            
            return output
            
        except Exception as e:
            print(f"GPT-4 API error: {e}")
            # Fall back to template-based response
            return generate_template_response(top_funds, industry, country, stage, description)
        
    except Exception as e:
        error_msg = f"Error searching funding institutions: {str(e)}"
        print(error_msg)
        return error_msg

def generate_template_response(funds: List[Dict], industry: str, country: str, stage: str, description: str) -> str:
    """Generate intelligent template-based response as fallback"""
    output = f"# 📈 INVESTMENT STRATEGY REPORT\n"
    output += f"**Company Profile:** {industry} | {country} | {stage}\n\n"
    
    # Intelligent Executive Summary
    output += "### 🎯 EXECUTIVE SUMMARY\n\n"
    market_analysis = generate_intelligent_market_analysis(industry, country, stage, len(funds))
    output += f"{market_analysis}\n\n"
    
    # Top Recommendations with detailed analysis
    output += "### 📊 TOP INVESTMENT FIRM RECOMMENDATIONS\n\n"
    
    for i, fund in enumerate(funds[:3], 1):
        output += f"**{i}. {fund['name']}**\n\n"
        
        # Firm Profile
        output += "**Firm Profile:**\n"
        output += f"- Location: {fund['country']}\n"
        output += f"- Investment Focus: {fund['areas']}\n"
        output += f"- Funding Stages: {fund['stage']}\n"
        output += f"- Typical Investment: {fund['funding_amount']}\n"
        output += f"- Investor Type: {fund['investor_type']}\n\n"
        
        # Intelligent matching analysis
        matching_analysis = generate_intelligent_matching_analysis(fund, industry, stage, description, country)
        output += f"**Why This Firm Matches:** {matching_analysis}\n\n"
        
        # Strategic approach
        strategic_approach = generate_intelligent_approach_strategy(fund, industry, description)
        output += f"**Strategic Approach:** {strategic_approach}\n\n"
    
    # Intelligent Strategic Insights
    output += "### 💡 KEY STRATEGIC INSIGHTS\n\n"
    strategic_insights = generate_intelligent_strategic_insights(funds[:3], industry, stage, description, country)
    output += f"{strategic_insights}\n\n"
    
    # Market Considerations
    output += "### ⚠️ MARKET CONSIDERATIONS\n\n"
    market_considerations = generate_intelligent_market_considerations(industry, country, stage)
    output += f"{market_considerations}\n\n"
    
    # Firm Profiles with Links
    output += "---\n\n### 🔗 INVESTMENT FIRM PROFILES\n\n"
    for fund in funds[:3]:
        firm_link = fund['link'] if fund['link'] and fund['link'] != 'nan' else "Contact information available upon request"
        output += f"**{fund['name']}**\n"
        output += f"📍 {fund['country']} | 💰 {fund['funding_amount']} | 🏷️ {fund['investor_type']}\n"
        output += f"🌐 {firm_link}\n\n"
    
    return output

def generate_intelligent_market_analysis(industry: str, country: str, stage: str, num_firms: int) -> str:
    """Generate intelligent market analysis based on specific parameters"""
    analyses = []
    
    # Industry-specific analysis
    if "AI" in industry.upper() or "artificial intelligence" in industry.lower():
        analyses.append(f"The {industry} sector is experiencing robust investor interest, particularly in {country}, with strong appetite for innovative applications.")
    elif "biotech" in industry.lower() or "bio" in industry.lower():
        analyses.append(f"Biotechnology investments in {country} are showing strong momentum, especially for companies with clinical validation.")
    elif "fintech" in industry.lower():
        analyses.append(f"FinTech remains a priority sector for {country} investors, with regulatory clarity driving increased investment activity.")
    else:
        analyses.append(f"The {industry} investment landscape in {country} presents selective but promising opportunities for differentiated companies.")
    
    # Stage-specific analysis
    if "pre-seed" in stage.lower():
        analyses.append(f"Pre-seed funding is highly competitive, but our analysis identifies {num_firms} firms with strong alignment to your profile.")
    elif "seed" in stage.lower():
        analyses.append(f"Seed-stage opportunities are abundant, with {num_firms} identified firms showing excellent strategic fit for your business model.")
    elif "series a" in stage.lower():
        analyses.append(f"Series A funding requires strong traction metrics, and the {num_firms} firms identified have track records of supporting companies at your stage.")
    
    return " ".join(analyses)

def generate_intelligent_matching_analysis(fund: Dict, industry: str, stage: str, description: str, country: str) -> str:
    """Generate intelligent, specific matching analysis"""
    analysis_parts = []
    
    # Investment focus alignment
    fund_areas = fund['areas'].lower()
    if industry.lower() in fund_areas or any(keyword in fund_areas for keyword in industry.lower().split()):
        analysis_parts.append(f"This firm demonstrates strong alignment with your {industry} focus, as their investment portfolio specifically targets {fund['areas']}")
    else:
        # Find related areas
        if "ai" in industry.lower() and ("tech" in fund_areas or "software" in fund_areas):
            analysis_parts.append(f"While not exclusively {industry}-focused, their {fund['areas']} portfolio includes technology companies that would appreciate your innovation")
    
    # Stage alignment
    fund_stages = fund['stage'].lower()
    if stage.lower() in fund_stages:
        analysis_parts.append(f"Their {fund['stage']} investment mandate directly matches your current {stage} funding requirements")
    else:
        # Check for stage compatibility
        stage_keywords = [s.strip() for s in stage.lower().replace(',', ' ').split()]
        if any(sk in fund_stages for sk in stage_keywords):
            analysis_parts.append(f"Their funding stages ({fund['stage']}) include your target {stage} investment level")
    
    # Geographic considerations
    if fund['country'].lower() == country.lower() or country.lower() in fund['country'].lower():
        analysis_parts.append(f"Being based in {fund['country']}, they offer local market expertise and regulatory knowledge that could be valuable for your expansion")
    elif "international" in fund['country'].lower() or fund['country'] == "Unknown":
        analysis_parts.append(f"Their international investment scope provides access to global markets and cross-border opportunities")
    
    # Description-based matching
    if description:
        desc_lower = description.lower()
        if "oxford" in desc_lower and "university" in fund.get('investor_type', '').lower():
            analysis_parts.append("Your Oxford University background aligns well with their focus on university spin-outs and academic commercialization")
        elif "clinical" in desc_lower and ("healthcare" in fund_areas or "bio" in fund_areas):
            analysis_parts.append("Your clinical validation progress resonates with their healthcare investment thesis and due diligence requirements")
        elif "deep learning" in desc_lower and "ai" in fund_areas:
            analysis_parts.append("Your deep learning expertise aligns perfectly with their AI investment focus and technical evaluation capabilities")
    
    # Investment amount consideration
    if fund['funding_amount'] and fund['funding_amount'] != 'N/A':
        analysis_parts.append(f"The typical investment range of {fund['funding_amount']} appears suitable for your current funding objectives and company valuation stage")
    
    # Combine all parts into coherent analysis
    if analysis_parts:
        return ". ".join(analysis_parts) + "."
    else:
        return f"This firm represents a strategic opportunity based on their {fund['investor_type']} approach and {fund['areas']} investment focus, which could provide valuable industry connections and growth capital for your {stage} funding round."

def generate_intelligent_approach_strategy(fund: Dict, industry: str, description: str) -> str:
    """Generate specific, intelligent approach strategies"""
    strategies = []
    
    # Strategy based on investor type
    investor_type = fund.get('investor_type', '').lower()
    if "university" in investor_type:
        strategies.append("Leverage your academic credentials and research background - request introduction through Oxford's commercialization office or alumni network")
        strategies.append("Prepare a research-focused pitch emphasizing IP portfolio and scientific validation rather than just market metrics")
    elif "accelerator" in investor_type:
        strategies.append("Apply through their formal program process, emphasizing team coachability and willingness to iterate based on market feedback")
        strategies.append("Attend their demo days or networking events to build relationships before formal application")
    elif "corporate" in investor_type:
        strategies.append("Focus on strategic partnership potential and synergies with their parent company's business objectives")
        strategies.append("Prepare case studies showing how your technology could enhance their existing portfolio companies")
    else:
        # Standard VC approach
        strategies.append("Research their recent portfolio additions and request warm introductions through mutual connections or portfolio companies")
        strategies.append("Prepare detailed market analysis and competitive positioning to demonstrate thorough market understanding")
    
    # Industry-specific strategies
    if "bio" in industry.lower() and "clinical" in description.lower():
        strategies.append("Emphasize your clinical validation progress and regulatory pathway clarity, as these are key decision factors for biotech investors")
    elif "ai" in industry.lower():
        strategies.append("Prepare a compelling technical demo and be ready to discuss your AI model's proprietary advantages and data moat")
    
    return ". ".join(strategies[:3]) + "."  # Limit to top 3 strategies

def generate_intelligent_strategic_insights(funds: List[Dict], industry: str, stage: str, description: str, country: str) -> str:
    """Generate intelligent strategic insights specific to this situation"""
    insights = []
    
    # What makes this company attractive
    attractions = []
    if "oxford" in description.lower():
        attractions.append("strong academic pedigree from Oxford University")
    if "clinical" in description.lower():
        attractions.append("clinical validation reducing technology risk")
    if "deep learning" in description.lower():
        attractions.append("advanced technical capabilities in high-demand field")
    
    if attractions:
        insights.append(f"Your company's key attractions to investors include: {', '.join(attractions)}, which directly address common investor concerns about team quality and technology risk.")
    
    # Potential investor concerns
    concerns = []
    if "early" in description.lower() or "pre-seed" in stage.lower():
        concerns.append("early-stage execution risk")
    if "cancer" in description.lower():
        concerns.append("long regulatory timeline for medical devices")
    if not any(market_word in description.lower() for market_word in ["customer", "revenue", "sales", "pilot"]):
        concerns.append("limited market traction evidence")
    
    if concerns:
        insights.append(f"Potential investor concerns may include: {', '.join(concerns)}. Address these proactively by emphasizing your risk mitigation strategies and market validation progress.")
    
    # Market positioning
    positioning_advice = f"Position your company as a {industry} innovator with proven technology capabilities and clear commercial pathway in the {country} market."
    if len(funds) > 2:
        positioning_advice += f" With {len(funds)} potential investors identified, you have leverage to be selective about strategic fit and terms."
    insights.append(positioning_advice)
    
    # Alternative strategies
    alt_strategies = []
    if "biotech" in industry.lower():
        alt_strategies.append("government innovation grants for healthcare technology")
        alt_strategies.append("pharmaceutical company strategic partnerships")
    elif "ai" in industry.lower():
        alt_strategies.append("corporate venture arms of tech companies")
        alt_strategies.append("government AI development initiatives")
    else:
        alt_strategies.append("strategic corporate partnerships")
        alt_strategies.append("revenue-based financing for growth capital")
    
    insights.append(f"Alternative funding strategies to consider: {', '.join(alt_strategies[:2])}, which could complement or substitute for traditional VC funding.")
    
    return " ".join(insights)

def generate_intelligent_market_considerations(industry: str, country: str, stage: str) -> str:
    """Generate specific market considerations"""
    considerations = []
    
    # Industry-specific market conditions
    if "ai" in industry.lower():
        considerations.append(f"AI investment activity in {country} remains strong despite broader market conditions, with particular interest in applied AI solutions with clear ROI.")
    elif "biotech" in industry.lower():
        considerations.append(f"Biotech funding in {country} is showing resilience, especially for companies with de-risked clinical programs and clear regulatory pathways.")
    else:
        considerations.append(f"{industry} sector investment activity in {country} is selective but active for companies with strong fundamentals.")
    
    # Stage-specific timing
    if "pre-seed" in stage.lower():
        considerations.append("Pre-seed funding typically requires 3-6 months to complete, with emphasis on team and technology validation.")
    elif "seed" in stage.lower():
        considerations.append("Seed rounds generally take 4-8 months to close, requiring demonstrated product-market fit indicators.")
    elif "series a" in stage.lower():
        considerations.append("Series A funding cycles average 6-12 months, with investors requiring strong revenue growth and market traction metrics.")
    
    # Market trends
    if country.upper() in ["UK", "UNITED KINGDOM"]:
        considerations.append("UK investor appetite remains strong for deep tech companies, supported by government innovation policies and tax incentives.")
    
    considerations.append("Current market conditions favor companies with clear value propositions, strong unit economics, and experienced management teams.")
    
    return " ".join(considerations)

def generate_mock_results(industry: str, country: str, stage: str, description: str) -> str:
    """Generate enhanced mock results for testing with professional format"""
    
    mock_funds = []
    
    if "UK" in country.upper() or "United Kingdom" in country:
        if "AI" in industry.upper() or "artificial intelligence" in industry.lower():
            mock_funds = [
                {
                    "name": "Oxford Sciences Innovation",
                    "stage": "Pre-Seed, Seed, Series A",
                    "areas": "AI, Deep Tech, Oxford Spinouts",
                    "country": "United Kingdom",
                    "investor_type": "University Venture Capital",
                    "funding_amount": "£100K - £2M",
                    "link": "https://www.oxfordsciencesinnovation.com"
                },
                {
                    "name": "Entrepreneur First",
                    "stage": "Pre-Seed, Seed",
                    "areas": "AI, Machine Learning, Deep Tech",
                    "country": "United Kingdom",
                    "investor_type": "Accelerator/VC",
                    "funding_amount": "£100K - £500K",
                    "link": "https://www.joinef.com"
                },
                {
                    "name": "Amadeus Capital Partners",
                    "stage": "Seed, Series A, Series B",
                    "areas": "AI, Healthcare Tech, Enterprise Software",
                    "country": "United Kingdom", 
                    "investor_type": "Venture Capital",
                    "funding_amount": "£1M - £10M",
                    "link": "https://www.amadeuscapital.com"
                }
            ]
    
    # Default mock data if no specific matches
    if not mock_funds:
        mock_funds = [
            {
                "name": f"{country} Innovation Capital",
                "stage": stage,
                "areas": f"{industry}, Technology Innovation",
                "country": country,
                "investor_type": "Venture Capital",
                "funding_amount": "Stage-appropriate investment",
                "link": "https://example-innovation-capital.com"
            },
            {
                "name": f"Global {industry} Ventures",
                "stage": stage,
                "areas": f"{industry}, Emerging Technologies",
                "country": "International",
                "investor_type": "Sector-focused VC",
                "funding_amount": "Series-based investment",
                "link": "https://global-sector-ventures.com"
            }
        ]
    
    # Use professional template format
    output = f"# 📈 INVESTMENT STRATEGY REPORT\n"
    output += f"**Company Profile:** {industry} | {country} | {stage}\n\n"
    output += "*Note: This is a demonstration report with sample data. In production, this connects to our comprehensive investment database.*\n\n"
    
    # Executive Summary
    output += "### 🎯 EXECUTIVE SUMMARY\n\n"
    output += f"Our analysis of the {industry} investment ecosystem in {country} reveals strong funding opportunities for {stage} companies. The following {len(mock_funds)} investment firms demonstrate exceptional alignment with your company profile and strategic objectives.\n\n"
    
    # Top Recommendations
    output += "### 📊 TOP INVESTMENT FIRM RECOMMENDATIONS\n\n"
    
    for i, fund in enumerate(mock_funds, 1):
        output += f"#### **{i}. {fund['name']}**\n\n"
        output += f"- **Investment Focus Alignment:** {fund['areas']} - Direct strategic fit with {industry} sector\n"
        output += f"- **Stage Compatibility:** {fund['stage']} - Perfect alignment with your {stage} funding requirements\n"
        output += f"- **Strategic Value-Add:** {generate_mock_value_add(fund, industry)}\n"
        output += f"- **Approach Strategy:** {generate_mock_approach_strategy(fund, industry)}\n"
        output += f"- **Success Probability:** High - Strong sector focus and stage alignment\n\n"
    
    # Strategic Recommendations
    output += "### 💡 STRATEGIC RECOMMENDATIONS\n\n"
    output += "1. **Optimal Timing:** Q2-Q3 typically offers highest investment activity and partner availability\n"
    output += "2. **Preparation Checklist:** Financial models, competitive analysis, market sizing, team credentials\n"
    output += f"3. **Positioning Strategy:** Emphasize {industry} differentiation and {country} market dynamics\n"
    output += "4. **Alternative Options:** Consider strategic investors and government innovation grants\n\n"
    
    # Key Considerations
    output += "### ⚠️ KEY CONSIDERATIONS\n\n"
    output += f"- {industry} sector showing strong investor interest in current market conditions\n"
    output += f"- {stage} funding cycles typically require 4-8 months for completion\n"
    output += "- Market validation and traction metrics are critical success factors\n\n"
    
    # Firm Profiles with Links
    output += "---\n\n### 🔗 INVESTMENT FIRM PROFILES\n\n"
    for fund in mock_funds:
        output += f"**{fund['name']}**\n"
        output += f"📍 {fund['country']} | 💰 {fund['funding_amount']} | 🏷️ {fund['investor_type']}\n"
        output += f"🌐 {fund['link']}\n\n"
    
    return output

def generate_mock_value_add(fund: dict, industry: str) -> str:
    """Generate mock strategic value-add descriptions"""
    if "Oxford" in fund['name']:
        return "Access to Oxford University research network and world-class technical talent pipeline"
    elif "Entrepreneur First" in fund['name']:
        return "Intensive acceleration program with global network of technical co-founders"
    elif "Global" in fund['name']:
        return f"International market access and strategic partnerships in {industry} ecosystem"
    else:
        return f"Deep sector expertise and extensive network in {industry} market"

def generate_mock_approach_strategy(fund: dict, industry: str) -> str:
    """Generate mock approach strategies"""
    if "University" in fund.get('investor_type', ''):
        return "Leverage academic connections and emphasize research-to-market commercialization story"
    elif "Accelerator" in fund.get('investor_type', ''):
        return "Apply through competitive program focusing on team dynamics and coachability"
    else:
        return f"Research recent {industry} portfolio additions and request warm introduction through mutual connections"

def generate_simple_reason(fund: dict, industry: str, stage: str, description: str) -> str:
    """Generate simple recommendation reasoning"""
    reasons = []
    
    if stage.lower() in fund['stage'].lower():
        reasons.append(f"Specializes in {stage} stage investments")
    
    if industry.lower() in fund['areas'].lower():
        reasons.append(f"Extensive experience in {industry} sector")
        
    if "oxford" in description.lower() or "cambridge" in description.lower():
        reasons.append("Values teams with top-tier academic backgrounds")
    
    if "AI" in industry and "AI" in fund['areas']:
        reasons.append("Strong focus on artificial intelligence and machine learning")
    
    if not reasons:
        reasons.append("Investment philosophy aligns well with company development goals")
    
    return "; ".join(reasons[:2])

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    import re
    
    text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    words = cleaned_text.split()
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'our', 'we', 'us', 'they', 
        'them', 'their', 'this', 'that', 'these', 'those', 'from', 'up', 'down', 'out', 'off', 
        'over', 'under', 'again', 'further', 'then', 'once', 'company', 'companies', 'business'
    }
    
    keywords = []
    for word in words:
        if len(word) > 2 and word not in stop_words:
            keywords.append(word)
    
    return list(set(keywords))

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

def generate_recommendation_reason(fund: Dict, industry: str, stage: str, description: str) -> str:
    """Generate sophisticated recommendation reasoning"""
    reasons = []
    
    # Stage matching
    stage_keywords = extract_keywords(stage)
    fund_stage_keywords = extract_keywords(fund['stage'])
    
    for sk in stage_keywords:
        for fsk in fund_stage_keywords:
            if sk in fsk or fsk in sk:
                reasons.append(f"Focuses on {fund['stage']} stage investments")
                break
        if reasons:
            break
    
    # Industry matching
    industry_keywords = extract_keywords(industry)
    fund_areas_keywords = extract_keywords(fund['areas'])
    
    matched_areas = []
    for ik in industry_keywords:
        for fak in fund_areas_keywords:
            if ik in fak or fak in ik:
                matched_areas.append(ik)
    
    if matched_areas:
        areas_str = ", ".join(list(set(matched_areas))[:2])
        reasons.append(f"Strong track record in {areas_str} and related sectors")
    
    # Description-based intelligent matching
    description_keywords = extract_keywords(description)
    common_keywords = []
    for dk in description_keywords:
        for fak in fund_areas_keywords:
            if len(dk) > 2 and (dk in fak or fak in dk):
                common_keywords.append(dk)
    
    if common_keywords:
        unique_keywords = list(set(common_keywords))[:2]
        keywords_str = ", ".join(unique_keywords)
        reasons.append(f"Specialized expertise in {keywords_str} technologies")
    
    # Overall similarity assessment
    overall_similarity = calculate_text_similarity(
        f"{industry} {description}", 
        f"{fund['areas']} {fund['stage']}"
    )
    
    if overall_similarity > 0.3 and not reasons:
        reasons.append("Investment focus highly aligned with company business model")
    elif overall_similarity > 0.5:
        reasons.append("Perfect strategic fit with company's development trajectory")
    
    # Geographic or additional factors
    if not reasons:
        if fund.get('country', '').lower() in description.lower():
            reasons.append(f"Deep market presence in {fund['country']}")
        else:
            reasons.append("Investment philosophy strongly aligned with company objectives")
    
    if not reasons:
        reasons.append("Relevant sector investment experience and network")
    
    return "; ".join(reasons[:2])

# Create tool with improved English description
SearchFundingTool = StructuredTool.from_function(
    name="SearchFundingTool", 
    description="Search for matching funding institutions and investors. Input: industry, country, funding stage, company description. Output: recommended investment firms list with detailed analysis and approach suggestions.",
    func=search_funding,
    args_schema=FundingQuery
)