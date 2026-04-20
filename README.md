# finance-toolkit
A growing collection of python tools for financial analysis, technical indicators, backtesting and algorithmic trading.
Built from Scratch - no black boxes

## Live Demo
https://steven-finance-dashboard.streamlit.app/

## What's inside

| File | Description |
|------|-------------|
| `day01_basics.py` | Market data loading, 52W stats |
| `day02_analysis.py` | Multi-stock returns, correlation, performance |
| `day03_charts.py` | Plotly charts, candlestick, heatmap |
| `indicators.py` | Custom TA library: SMA, EMA, RSI, MACD |
| `dashboard.py` | Full Streamlit dashboard with DCF calculator |
| `plot_utils.py` | Reusable chart functions |

## Whats being built
- [] Market data analysis
- [] Technical indicators library
- [] Backtetsting framework
- [] Streamlit dashboards
- [] Algo trading bots 

## Stack
Python · pandas · yfinance · Streamlit · Plotly · Matplotlib

## Features
- Interactive candlestick charts with volume
- Custom technical indicators (SMA, EMA, RSI, MACD)
- Peer comparison with normalized performance
- DCF valuation model with adjustable parameters
- CSV export with calculated indicators


## Setup
pip install -r requirements.txt
streamlit run dashboard.py

## About
Built as part of an 84-day Finance × Python masterplan.
Documenting the journey publicly on 