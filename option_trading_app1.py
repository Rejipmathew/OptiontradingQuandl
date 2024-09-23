# option_trading_app.py

import streamlit as st
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import os

# Configure the page
st.set_page_config(page_title="Option Trading App", layout="wide")

# Title
st.title("📈 Option Trading App with Quandl")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

def user_input_ticker():
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    return ticker.upper()

def user_input_quandl_api_key():
    api_key = st.sidebar.text_input("9uD8246Anu3wcL-x8C1A")
    return api_key

def get_stock_data(ticker, api_key):
    """
    Fetch historical stock data from Quandl.
    """
    try:
        # Example using WIKI dataset (Note: WIKI dataset has been deprecated; replace with an active dataset)
        # You need to replace 'WIKI/' with the appropriate dataset code.
        # For example, 'EOD/' for end-of-day data.
        data = quandl.get(f"EOD/{ticker}", authtoken=api_key)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def get_option_chain(ticker, expiration, api_key):
    """
    Fetch option chain data from Quandl.
    Note: Requires access to a dataset that provides option chains.
    Replace 'OPTIONMETRICS/' with the appropriate dataset code.
    """
    try:
        # Placeholder for option chain fetching
        # Example: OptionMetrics dataset (premium)
        # You need to adjust the dataset code and query based on your Quandl subscription
        option_calls = quandl.get(f"OPTIONMETRICS/{ticker}_CALLS_{expiration}", authtoken=api_key)
        option_puts = quandl.get(f"OPTIONMETRICS/{ticker}_PUTS_{expiration}", authtoken=api_key)
        return option_calls, option_puts
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return pd.DataFrame(), pd.DataFrame()

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price
    S: Current stock price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free interest rate
    sigma: Volatility of the underlying stock
    option_type: 'call' or 'put'
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma **2 ) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price
    except Exception as e:
        st.error(f"Error in Black-Scholes calculation: {e}")
        return np.nan

# Main content
ticker = user_input_ticker()
api_key = user_input_quandl_api_key()

if ticker and api_key:
    # Set Quandl API Key
    quandl.ApiConfig.api_key = api_key

    # Fetch stock data
    stock_data = get_stock_data(ticker, api_key)
    
    if not stock_data.empty:
        # Get the latest closing price
        current_price = stock_data['Adj_Close'].iloc[-1]
        st.subheader(f"📊 Current Price of {ticker}: ${current_price:.2f}")
    else:
        current_price = None

    # Placeholder for fetching option expirations
    # Since option chain fetching depends on the dataset, you need to adjust accordingly
    expirations = st.sidebar.text_input("Option Expiration Date (YYYY-MM-DD)", "2024-12-20")
    
    if expirations:
        # Fetch option chains
        option_calls, option_puts = get_option_chain(ticker, expirations, api_key)
        
        if not option_calls.empty and not option_puts.empty:
            st.subheader(f"🔍 Options Chain for {ticker} - {expirations}")
            
            # Display calls and puts
            tab1, tab2 = st.tabs(["Calls", "Puts"])
            
            with tab1:
                st.dataframe(option_calls)
            with tab2:
                st.dataframe(option_puts)
            
            # Option selection
            st.sidebar.subheader("Select Option Parameters")
            option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
            if option_type == "call":
                options = option_calls
            else:
                options = option_puts
            
            if not options.empty:
                strike = st.sidebar.selectbox("Strike Price", sorted(options['Strike'].unique()))
                
                # Get option details
                option = options[options['Strike'] == strike].iloc[0]
                bid = option['Bid']
                ask = option['Ask']
                last_price = option['Last']
                volume = option['Volume']
                open_interest = option['Open_Interest']
                
                st.write(f"### Selected Option: {option_type.capitalize()} {strike}")
                st.write(f"**Bid:** ${bid} | **Ask:** ${ask} | **Last Price:** ${last_price}")
                st.write(f"**Volume:** {volume} | **Open Interest:** {open_interest}")
                
                # Black-Scholes Parameters
                st.sidebar.subheader("Black-Scholes Parameters")
                risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=1.5) / 100
                volatility = st.sidebar.number_input("Volatility (%)", value=20.0) / 100
                today = datetime.today()
                expiration_date = datetime.strptime(expirations, "%Y-%m-%d")
                T = (expiration_date - today).days / 365
                if T <= 0:
                    st.error("Expiration date must be in the future.")
                    T = 0.01  # Prevent division by zero
                
                # Calculate Black-Scholes price
                bs_price = black_scholes(S=current_price, K=strike, T=T, r=risk_free_rate, sigma=volatility, option_type=option_type)
                st.write(f"**Black-Scholes {option_type.capitalize()} Price:** ${bs_price:.2f}")
                
                # Plot Payoff
                st.subheader("Option Payoff Diagram")
                # Define range for underlying price
                S = np.linspace(current_price * 0.5, current_price * 1.5, 100)
                if option_type == 'call':
                    payoff = np.maximum(S - strike, 0) - bs_price
                else:
                    payoff = np.maximum(strike - S, 0) - bs_price
                
                fig, ax = plt.subplots()
                ax.plot(S, payoff, label='Payoff')
                ax.axhline(0, color='black', lw=0.5)
                ax.axvline(current_price, color='red', linestyle='--', label='Current Price')
                ax.set_xlabel('Stock Price at Expiration ($)')
                ax.set_ylabel('Profit / Loss ($)')
                ax.set_title(f'{option_type.capitalize()} Option Payoff')
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("No option data available for the selected strike price.")
        else:
            st.warning("No option data available for the selected expiration date or dataset.")
else:
    st.info("Please enter both a valid stock ticker symbol and your Quandl API Key.")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name](https://www.example.com)")
