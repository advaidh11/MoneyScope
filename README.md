# MoneyScope: AI-Powered Currency Pair Analysis

## Project Overview

MoneyScope is a comprehensive web application that provides real-time foreign exchange (forex) analysis and insights using artificial intelligence. Built with Streamlit and enhanced by the powerful Llama3 language model via Groq's API, this tool delivers professional-grade currency pair analysis that was previously only available to financial institutions.

## Features

- **Real-Time Exchange Rates**: Get up-to-date exchange rates between any two currencies from a selection of major and emerging market currencies.

- **Quick Currency Converter**: Easily convert amounts between currency pairs with the latest rates.

- **AI-Powered Analysis**: Generate comprehensive analysis reports for any currency pair including:
  - Current exchange rate significance
  - Recent financial news impact
  - Market trend analysis and sentiment
  - Short-term price movement predictions
  - Professional investment insights

- **Interactive User Interface**: Clean, intuitive design with progress tracking and downloadable reports.

- **LangGraph Workflow**: Structured analysis process that combines real-world data with AI reasoning.

## Target Users

MoneyScope is designed for:

- Individual forex traders and investors
- Business professionals dealing with international transactions
- Financial analysts and advisors
- Anyone interested in understanding currency market dynamics
- Students learning about foreign exchange markets

## Installation Requirements

### Prerequisites

- Python 3.8 or higher
- API keys for the following services:
  - Exchange Rate API (from exchangerate-api.com)
  - News API (from newsapi.org)
  - Groq API (from groq.com)

### Required Libraries

```
pip install streamlit
pip install requests
pip install pandas
pip install python-dotenv
pip install groq
pip install langgraph
pip install plotly
```

### Environment Setup

1. Clone the repository
2. Create a `.env` file in the root directory
3. Add your API keys to the `.env` file:
   ```
   EXCHANGE_RATE_API_KEY=your_exchange_rate_api_key
   NEWS_API_KEY=your_news_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Running the Application

Execute the following command in the project directory:

```
streamlit run app.py
```

The application will launch in your default web browser.

## Usage Guide

1. **Select Currency Pair**:
   - Choose a base currency and target currency from the dropdown menus
   - The application displays major currencies like USD, EUR, GBP as well as emerging market currencies

2. **Use the Quick Converter**:
   - Enter an amount in the base currency
   - See the converted amount in the target currency using the latest exchange rates

3. **Generate Analysis**:
   - Click the "Analyze Currency Pair" button
   - Watch as the application collects data, processes it, and generates insights
   - Review the comprehensive report with market analysis and predictions

4. **Save Reports**:
   - Download the analysis report as a Markdown file for future reference
   - Reports are timestamped and include all relevant data and insights

## How It Works

MoneyScope uses a LangGraph workflow to:

1. Collect exchange rate data from a reliable API
2. Gather recent news articles related to the currency pair
3. Analyze market trends and sentiment
4. Process this data using the Llama3 model via Groq
5. Generate a comprehensive, human-readable report with actionable insights
