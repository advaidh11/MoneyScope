import os
import json
import time
from datetime import datetime
import requests
import streamlit as st
from typing import Dict, List, Any, TypedDict
from groq import Groq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="MoneyScope Analysis Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Tool Functions ---
def get_exchange_rate(base_currency, target_currency):
    """Get the current exchange rate between two currencies."""
    api_key = os.getenv("EXCHANGE_RATE_API_KEY", "")
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "base_currency": base_currency,
                "target_currency": target_currency,
                "rate": data["conversion_rate"],
                "last_updated": data["time_last_update_utc"],
                "next_update": data["time_next_update_utc"],
                "status": "success"
            }
        return {"status": "error", "message": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_forex_news(currency_pair):
    """Get the latest news related to a specific currency pair."""
    base, target = currency_pair.split('/')
    api_key = os.getenv("NEWS_API_KEY", "")
    url = "https://newsapi.org/v2/everything"
    query = f"forex {base} {target} exchange rate"
    
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 3,
        "apiKey": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("name"),
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url"),
                    "description": article.get("description")
                }
                for article in data.get("articles", [])
            ]
        return {"status": "error", "message": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def analyze_forex_trends(currency_pair):
    """Simplified trend analysis for a currency pair."""
    import random
    base, target = currency_pair.split('/')
    trend_directions = ["upward", "downward", "sideways"]
    sentiment_options = ["bullish", "bearish", "neutral"]
    
    return {
        "base_currency": base,
        "target_currency": target,
        "current_trend": random.choice(trend_directions),
        "trend_strength": random.randint(3, 8),
        "public_sentiment": random.choice(sentiment_options),
        "volatility": round(random.uniform(0.1, 5.0), 2),
        "trading_volume": f"{random.randint(50, 500)}M"
    }

# --- LangGraph State Definition ---
class ForexAnalysisState(TypedDict):
    currency_pair: str
    exchange_rate_data: Dict
    news_data: List[Dict]
    trend_data: Dict
    data_analysis: str
    final_report: str

# --- LangGraph Nodes ---
def collect_exchange_rate(state: ForexAnalysisState) -> ForexAnalysisState:
    """Node to collect exchange rate data."""
    base_currency, target_currency = state["currency_pair"].split('/')
    exchange_rate_data = get_exchange_rate(base_currency, target_currency)
    return {"exchange_rate_data": exchange_rate_data}

def collect_news(state: ForexAnalysisState) -> ForexAnalysisState:
    """Node to collect news data."""
    news_data = get_forex_news(state["currency_pair"])
    return {"news_data": news_data}

def collect_trends(state: ForexAnalysisState) -> ForexAnalysisState:
    """Node to collect trend data."""
    trend_data = analyze_forex_trends(state["currency_pair"])
    return {"trend_data": trend_data}

def analyze_data(state: ForexAnalysisState) -> ForexAnalysisState:
    """Node to analyze collected data."""
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY", "")
    )
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    prompt = f"""
    You are a Forex Data Analyst specializing in analyzing currency pairs.
    
    Please analyze the following data for the currency pair {state['currency_pair']} as of {current_date}:
    
    1. Exchange Rate Data:
    {json.dumps(state['exchange_rate_data'], indent=2)}
    
    2. Recent News:
    {json.dumps(state['news_data'], indent=2)}
    
    3. Market Trends:
    {json.dumps(state['trend_data'], indent=2)}
    
    Provide a detailed analysis including:
    - Significance of the current rate
    - Impact of recent news on the currency pair
    - Analysis of current market trends and sentiment
    - Factors that might influence short-term movement
    
    Keep your analysis factual, insightful and focused on the data provided.
    """
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # You can choose the appropriate model
        messages=[
            {"role": "system", "content": "You are an expert Forex Data Analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    analysis = response.choices[0].message.content
    return {"data_analysis": analysis}

def generate_report(state: ForexAnalysisState) -> ForexAnalysisState:
    """Node to generate final report."""
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY", "")
    )
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    prompt = f"""
    You are a Forex Report Generator who creates clear, concise reports that are easy to understand.
    
    Using the following analysis and data for the currency pair {state['currency_pair']} as of {current_date}, 
    create a comprehensive but easy-to-read report:
    
    ## DATA ANALYSIS:
    {state['data_analysis']}
    
    ## EXCHANGE RATE:
    {json.dumps(state['exchange_rate_data'], indent=2)}
    
    ## NEWS:
    {json.dumps(state['news_data'], indent=2)}
    
    ## TRENDS:
    {json.dumps(state['trend_data'], indent=2)}
    
    Your report should include:
    1. A clear title and introduction
    2. Current exchange rate with time of update
    3. Summary of key news (as bullet points)
    4. Analysis of current trends and sentiment
    5. Short-term prediction (1 week outlook)
    6. Conclusion with key takeaways
    
    Format your report with Markdown headings, bullet points, and clear sections.
    The report should be professional but accessible to non-experts.
    """
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # You can choose the appropriate model
        messages=[
            {"role": "system", "content": "You are an expert Forex Report Generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    report = response.choices[0].message.content
    return {"final_report": report}

# --- LangGraph Workflow ---
class ForexAnalysisGraph:
    """A workflow for analyzing forex currency pairs using LangGraph."""
    
    def __init__(self):
        """Initialize the graph workflow."""
        # Build the graph
        workflow = StateGraph(ForexAnalysisState)
        
        # Add nodes
        workflow.add_node("collect_exchange_rate", collect_exchange_rate)
        workflow.add_node("collect_news", collect_news)
        workflow.add_node("collect_trends", collect_trends)
        workflow.add_node("analyze_data", analyze_data)
        workflow.add_node("generate_report", generate_report)
        
        # Define edges (linear workflow)
        workflow.add_edge("collect_exchange_rate", "collect_news")
        workflow.add_edge("collect_news", "collect_trends")
        workflow.add_edge("collect_trends", "analyze_data")
        workflow.add_edge("analyze_data", "generate_report")
        workflow.add_edge("generate_report", END)
        
        # Set the entry point
        workflow.set_entry_point("collect_exchange_rate")
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def analyze_currency_pair(self, currency_pair):
        """Analyze a currency pair and generate a report."""
        if '/' not in currency_pair:
            return "Error: Currency pair should be in format 'XXX/YYY' (e.g., 'USD/INR')"
        
        # Initialize state
        initial_state = {"currency_pair": currency_pair}
        
        # Execute the graph
        result = self.graph.invoke(initial_state)
        
        # Return the final report
        return result["final_report"]

# --- Streamlit App ---
def main():
    """Main function for the Streamlit app."""
    st.title("💹 MoneyScope - Currency Pair Analysis")
    
    with st.container():
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin-top: 0;">What This App Does</h3>
            <p>This app analyzes forex currency pairs in real-time using AI and provides:</p>
            <ul>
                <li>Current exchange rates with latest updates</li>
                <li>Relevant financial news analysis</li>
                <li>Market trend insights and sentiment</li>
                <li>Short-term price movement predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Currency selection with improved UI
    st.subheader("Select Currency Pair")
    major_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
    emerging_currencies = ["INR", "CNY", "BRL", "ZAR", "MXN", "RUB", "TRY", "SGD"]
    all_currencies = sorted(list(set(major_currencies + emerging_currencies)))
    
    col1, col2 = st.columns(2)
    with col1:
        base_currency = st.selectbox(
            "Base Currency", 
            options=all_currencies, 
            index=all_currencies.index("USD"),
            help="The currency you're converting from"
        )
    with col2:
        target_options = [c for c in all_currencies if c != base_currency]
        default_target = "INR" if "INR" in target_options else target_options[0]
        target_currency = st.selectbox(
            "Target Currency", 
            options=target_options, 
            index=target_options.index(default_target),
            help="The currency you're converting to"
        )
    
    currency_pair = f"{base_currency}/{target_currency}"
    
    # Quick converter in a nice card
    st.markdown("""
    <div style="background-color: #e9f7fe; padding: 15px; border-radius: 10px; border-left: 5px solid #4dabf7;">
        <h3 style="margin-top: 0;">Quick Converter</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input(
            f"Amount in {base_currency}", 
            min_value=0.01, 
            value=100.0, 
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        # Get rate for converter
        try:
            rate_data = get_exchange_rate(base_currency, target_currency)
            if rate_data.get("status") == "success":
                rate = rate_data.get("rate", 0)
                converted = amount * rate
                st.markdown(f"""
                <div style="background-color: #edf8ee; padding: 10px; border-radius: 5px; text-align: center;">
                    <h2 style="margin: 0;">{amount:.2f} {base_currency} = {converted:.2f} {target_currency}</h2>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Rate: 1 {base_currency} = {rate} {target_currency}")
                st.caption(f"Last updated: {rate_data.get('last_updated', 'Unknown')}")
            else:
                st.error("Could not fetch exchange rate")
        except Exception as e:
            st.error(f"Error in conversion: {str(e)}")
    
    # Check API keys
    missing_keys = []
    if not os.getenv("EXCHANGE_RATE_API_KEY"):
        missing_keys.append("Exchange Rate API Key")
    if not os.getenv("NEWS_API_KEY"):
        missing_keys.append("News API Key")
    if not os.getenv("GROQ_API_KEY"):
        missing_keys.append("Groq API Key")
    
    if missing_keys:
        st.warning(f"Missing API keys: {', '.join(missing_keys)}. Please add them in your .env file.")
    
    analyze_button = st.button(
        "🔍 Analyze Currency Pair", 
        type="primary", 
        disabled=bool(missing_keys),
        help="Click to generate a comprehensive analysis report"
    )
    
    if analyze_button:
        progress_container = st.empty()
        result_container = st.empty()
        
        with progress_container.container():
            st.subheader("📊 Analysis in Progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define workflow steps
            steps = [
                "Collecting Exchange Rate Data", 
                "Gathering Financial News", 
                "Analyzing Market Trends",
                "Processing Data Analysis",
                "Generating Final Report"
            ]
            
            # Show progress
            for i, step in enumerate(steps):
                progress_value = (i + 1) * (100 // len(steps))
                progress_bar.progress(progress_value)
                status_text.text(f"Step {i+1}/{len(steps)}: {step}")
                time.sleep(0.8)  # Simulate processing time
            
            try:
                status_text.text("Finalizing analysis...")
                # Run the LangGraph workflow
                forex_graph = ForexAnalysisGraph()
                result = forex_graph.analyze_currency_pair(currency_pair)
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return
        
        progress_container.empty()
        
        with result_container.container():
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 5px 15px; border-radius: 10px; margin-bottom: 10px;">
                <h2 style="margin-top: 10px;">Analysis Results: {currency_pair}</h2>
                <p><strong>Report Generated:</strong> {datetime.now().strftime('%B %d, %Y %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the report in a nice format
            st.markdown(result)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=result,
                file_name=f"forex_analysis_{currency_pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()











# import os
# import json
# import time
# from datetime import datetime
# import requests
# import streamlit as st
# from typing import Dict, List, Any, TypedDict
# from langchain_google_genai import GoogleGenerativeAI
# from langgraph.graph import StateGraph, END
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set page configuration
# st.set_page_config(
#     page_title="Forex Analysis Dashboard",
#     page_icon="💹",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Tool Functions ---
# def get_exchange_rate(base_currency, target_currency):
#     """Get the current exchange rate between two currencies."""
#     api_key = os.getenv("EXCHANGE_RATE_API_KEY", "")
#     url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()
#             return {
#                 "base_currency": base_currency,
#                 "target_currency": target_currency,
#                 "rate": data["conversion_rate"],
#                 "last_updated": data["time_last_update_utc"],
#                 "next_update": data["time_next_update_utc"],
#                 "status": "success"
#             }
#         return {"status": "error", "message": f"API returned status code {response.status_code}"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# def get_forex_news(currency_pair):
#     """Get the latest news related to a specific currency pair."""
#     base, target = currency_pair.split('/')
#     api_key = os.getenv("NEWS_API_KEY", "")
#     url = "https://newsapi.org/v2/everything"
#     query = f"forex {base} {target} exchange rate"
    
#     params = {
#         "q": query,
#         "sortBy": "publishedAt",
#         "language": "en",
#         "pageSize": 3,
#         "apiKey": api_key
#     }
    
#     try:
#         response = requests.get(url, params=params)
#         if response.status_code == 200:
#             data = response.json()
#             return [
#                 {
#                     "title": article.get("title"),
#                     "source": article.get("source", {}).get("name"),
#                     "published_at": article.get("publishedAt"),
#                     "url": article.get("url"),
#                     "description": article.get("description")
#                 }
#                 for article in data.get("articles", [])
#             ]
#         return {"status": "error", "message": f"API returned status code {response.status_code}"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# def analyze_forex_trends(currency_pair):
#     """Simplified trend analysis for a currency pair."""
#     import random
#     base, target = currency_pair.split('/')
#     trend_directions = ["upward", "downward", "sideways"]
#     sentiment_options = ["bullish", "bearish", "neutral"]
    
#     return {
#         "base_currency": base,
#         "target_currency": target,
#         "current_trend": random.choice(trend_directions),
#         "trend_strength": random.randint(3, 8),
#         "public_sentiment": random.choice(sentiment_options),
#         "volatility": round(random.uniform(0.1, 5.0), 2),
#         "trading_volume": f"{random.randint(50, 500)}M"
#     }

# # --- LangGraph State Definition ---
# class ForexAnalysisState(TypedDict):
#     currency_pair: str
#     exchange_rate_data: Dict
#     news_data: List[Dict]
#     trend_data: Dict
#     data_analysis: str
#     final_report: str

# # --- LangGraph Nodes ---
# def collect_exchange_rate(state: ForexAnalysisState) -> ForexAnalysisState:
#     """Node to collect exchange rate data."""
#     base_currency, target_currency = state["currency_pair"].split('/')
#     exchange_rate_data = get_exchange_rate(base_currency, target_currency)
#     return {"exchange_rate_data": exchange_rate_data}

# def collect_news(state: ForexAnalysisState) -> ForexAnalysisState:
#     """Node to collect news data."""
#     news_data = get_forex_news(state["currency_pair"])
#     return {"news_data": news_data}

# def collect_trends(state: ForexAnalysisState) -> ForexAnalysisState:
#     """Node to collect trend data."""
#     trend_data = analyze_forex_trends(state["currency_pair"])
#     return {"trend_data": trend_data}

# def analyze_data(state: ForexAnalysisState) -> ForexAnalysisState:
#     """Node to analyze collected data."""
#     llm = GoogleGenerativeAI(
#         model="gemini-1.5-flash",  # Updated model name
#         google_api_key=os.getenv("GEMINI_API_KEY", ""),
#         temperature=0.3
#     )
    
#     current_date = datetime.now().strftime("%B %d, %Y")
    
#     prompt = f"""
#     You are a Forex Data Analyst specializing in analyzing currency pairs.
    
#     Please analyze the following data for the currency pair {state['currency_pair']} as of {current_date}:
    
#     1. Exchange Rate Data:
#     {json.dumps(state['exchange_rate_data'], indent=2)}
    
#     2. Recent News:
#     {json.dumps(state['news_data'], indent=2)}
    
#     3. Market Trends:
#     {json.dumps(state['trend_data'], indent=2)}
    
#     Provide a detailed analysis including:
#     - Significance of the current rate
#     - Impact of recent news on the currency pair
#     - Analysis of current market trends and sentiment
#     - Factors that might influence short-term movement
    
#     Keep your analysis factual, insightful and focused on the data provided.
#     """
    
#     analysis = llm.invoke(prompt).content
#     return {"data_analysis": analysis}

# def generate_report(state: ForexAnalysisState) -> ForexAnalysisState:
#     """Node to generate final report."""
#     llm = GoogleGenerativeAI(
#         model="gemini-1.5-flash",  # Updated model name
#         google_api_key=os.getenv("GEMINI_API_KEY", ""),
#         temperature=0.2
#     )
    
#     current_date = datetime.now().strftime("%B %d, %Y")
    
#     prompt = f"""
#     You are a Forex Report Generator who creates clear, concise reports that are easy to understand.
    
#     Using the following analysis and data for the currency pair {state['currency_pair']} as of {current_date}, 
#     create a comprehensive but easy-to-read report:
    
#     ## DATA ANALYSIS:
#     {state['data_analysis']}
    
#     ## EXCHANGE RATE:
#     {json.dumps(state['exchange_rate_data'], indent=2)}
    
#     ## NEWS:
#     {json.dumps(state['news_data'], indent=2)}
    
#     ## TRENDS:
#     {json.dumps(state['trend_data'], indent=2)}
    
#     Your report should include:
#     1. A clear title and introduction
#     2. Current exchange rate with time of update
#     3. Summary of key news (as bullet points)
#     4. Analysis of current trends and sentiment
#     5. Short-term prediction (1 week outlook)
#     6. Conclusion with key takeaways
    
#     Format your report with Markdown headings, bullet points, and clear sections.
#     The report should be professional but accessible to non-experts.
#     """
    
#     report = llm.invoke(prompt).content
#     return {"final_report": report}

# # --- LangGraph Workflow ---
# class ForexAnalysisGraph:
#     """A workflow for analyzing forex currency pairs using LangGraph."""
    
#     def __init__(self):
#         """Initialize the graph workflow."""
#         # Build the graph
#         workflow = StateGraph(ForexAnalysisState)
        
#         # Add nodes
#         workflow.add_node("collect_exchange_rate", collect_exchange_rate)
#         workflow.add_node("collect_news", collect_news)
#         workflow.add_node("collect_trends", collect_trends)
#         workflow.add_node("analyze_data", analyze_data)
#         workflow.add_node("generate_report", generate_report)
        
#         # Define edges (linear workflow)
#         workflow.add_edge("collect_exchange_rate", "collect_news")
#         workflow.add_edge("collect_news", "collect_trends")
#         workflow.add_edge("collect_trends", "analyze_data")
#         workflow.add_edge("analyze_data", "generate_report")
#         workflow.add_edge("generate_report", END)
        
#         # Set the entry point
#         workflow.set_entry_point("collect_exchange_rate")
        
#         # Compile the graph
#         self.graph = workflow.compile()
    
#     def analyze_currency_pair(self, currency_pair):
#         """Analyze a currency pair and generate a report."""
#         if '/' not in currency_pair:
#             return "Error: Currency pair should be in format 'XXX/YYY' (e.g., 'USD/INR')"
        
#         # Initialize state
#         initial_state = {"currency_pair": currency_pair}
        
#         # Execute the graph
#         result = self.graph.invoke(initial_state)
        
#         # Return the final report
#         return result["final_report"]

# # --- Streamlit App ---
# def main():
#     """Main function for the Streamlit app."""
#     st.title("💹 Forex Currency Pair Analysis")
    
#     with st.container():
#         st.markdown("""
#         <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
#             <h3 style="margin-top: 0;">What This App Does</h3>
#             <p>This app analyzes forex currency pairs in real-time using AI and provides:</p>
#             <ul>
#                 <li>Current exchange rates with latest updates</li>
#                 <li>Relevant financial news analysis</li>
#                 <li>Market trend insights and sentiment</li>
#                 <li>Short-term price movement predictions</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Currency selection with improved UI
#     st.subheader("Select Currency Pair")
#     major_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
#     emerging_currencies = ["INR", "CNY", "BRL", "ZAR", "MXN", "RUB", "TRY", "SGD"]
#     all_currencies = sorted(list(set(major_currencies + emerging_currencies)))
    
#     col1, col2 = st.columns(2)
#     with col1:
#         base_currency = st.selectbox(
#             "Base Currency", 
#             options=all_currencies, 
#             index=all_currencies.index("USD"),
#             help="The currency you're converting from"
#         )
#     with col2:
#         target_options = [c for c in all_currencies if c != base_currency]
#         default_target = "INR" if "INR" in target_options else target_options[0]
#         target_currency = st.selectbox(
#             "Target Currency", 
#             options=target_options, 
#             index=target_options.index(default_target),
#             help="The currency you're converting to"
#         )
    
#     currency_pair = f"{base_currency}/{target_currency}"
    
#     # Quick converter in a nice card
#     st.markdown("""
#     <div style="background-color: #e9f7fe; padding: 15px; border-radius: 10px; border-left: 5px solid #4dabf7;">
#         <h3 style="margin-top: 0;">Quick Converter</h3>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         amount = st.number_input(
#             f"Amount in {base_currency}", 
#             min_value=0.01, 
#             value=100.0, 
#             step=0.01,
#             format="%.2f"
#         )
    
#     with col2:
#         # Get rate for converter
#         try:
#             rate_data = get_exchange_rate(base_currency, target_currency)
#             if rate_data.get("status") == "success":
#                 rate = rate_data.get("rate", 0)
#                 converted = amount * rate
#                 st.markdown(f"""
#                 <div style="background-color: #edf8ee; padding: 10px; border-radius: 5px; text-align: center;">
#                     <h2 style="margin: 0;">{amount:.2f} {base_currency} = {converted:.2f} {target_currency}</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
#                 st.caption(f"Rate: 1 {base_currency} = {rate} {target_currency}")
#                 st.caption(f"Last updated: {rate_data.get('last_updated', 'Unknown')}")
#             else:
#                 st.error("Could not fetch exchange rate")
#         except Exception as e:
#             st.error(f"Error in conversion: {str(e)}")
    
#     # Check API keys
#     missing_keys = []
#     if not os.getenv("EXCHANGE_RATE_API_KEY"):
#         missing_keys.append("Exchange Rate API Key")
#     if not os.getenv("NEWS_API_KEY"):
#         missing_keys.append("News API Key")
#     if not os.getenv("GEMINI_API_KEY"):
#         missing_keys.append("Gemini API Key")
    
#     if missing_keys:
#         st.warning(f"Missing API keys: {', '.join(missing_keys)}. Please add them in your .env file.")
    
#     # Add a helpful explanation about model versions
#     st.info("This app uses Google's Gemini-1.5-Flash model for AI analysis. If you encounter model version errors, you may need to update the model name in the code to match the latest available version in your Gemini API account.")
    
#     analyze_button = st.button(
#         "🔍 Analyze Currency Pair", 
#         type="primary", 
#         disabled=bool(missing_keys),
#         help="Click to generate a comprehensive analysis report"
#     )
    
#     if analyze_button:
#         progress_container = st.empty()
#         result_container = st.empty()
        
#         with progress_container.container():
#             st.subheader("📊 Analysis in Progress...")
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Define workflow steps
#             steps = [
#                 "Collecting Exchange Rate Data", 
#                 "Gathering Financial News", 
#                 "Analyzing Market Trends",
#                 "Processing Data Analysis",
#                 "Generating Final Report"
#             ]
            
#             # Show progress
#             for i, step in enumerate(steps):
#                 progress_value = (i + 1) * (100 // len(steps))
#                 progress_bar.progress(progress_value)
#                 status_text.text(f"Step {i+1}/{len(steps)}: {step}")
#                 time.sleep(0.8)  # Simulate processing time
            
#             try:
#                 status_text.text("Finalizing analysis...")
#                 # Run the LangGraph workflow
#                 forex_graph = ForexAnalysisGraph()
#                 result = forex_graph.analyze_currency_pair(currency_pair)
#                 progress_bar.progress(100)
#                 status_text.text("Analysis complete!")
#                 time.sleep(0.5)
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#                 # Add troubleshooting assistance
#                 st.error("""
#                 Troubleshooting tips:
#                 1. Check if your Gemini API key is valid
#                 2. Verify that you have access to the Gemini 1.5 Flash model
#                 3. If the model name is different in your account, update lines 107 and 141 with the correct model name
#                 """)
#                 return
        
#         progress_container.empty()
        
#         with result_container.container():
#             st.markdown(f"""
#             <div style="background-color: #f8f9fa; padding: 5px 15px; border-radius: 10px; margin-bottom: 10px;">
#                 <h2 style="margin-top: 10px;">Analysis Results: {currency_pair}</h2>
#                 <p><strong>Report Generated:</strong> {datetime.now().strftime('%B %d, %Y %H:%M:%S')}</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Display the report in a nice format
#             st.markdown(result)
            
#             # Download button
#             st.download_button(
#                 label="📥 Download Report",
#                 data=result,
#                 file_name=f"forex_analysis_{currency_pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
#                 mime="text/markdown"
#             )

# if __name__ == "__main__":
#     main()