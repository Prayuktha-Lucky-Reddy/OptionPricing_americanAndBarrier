
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

st.title("Monte Carlo Option Pricing App")
st.markdown("Supports European and Asian Options ")


# Sidebar inputs
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Stock Symbol", "INFY.NS")
start_str = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2025-01-01")
end_str = st.sidebar.text_input("End Date (YYYY-MM-DD)", "2025-06-30")

start_date = pd.to_datetime(start_str)
end_date = pd.to_datetime(end_str)

option_type = st.sidebar.selectbox("Option Type", ["European", "Asian"])

#Downloading Infosys data from Jan 2025 to June 2025
data = yf.download("INFY.NS", start=start_date, end=end_date, auto_adjust=False)

#Calculate daily returns and volatility
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)
daily_vol = data['Return'].std()
vol = daily_vol * np.sqrt(252)

#Set option parameters
lcallt_price = data['Close'].iloc[-1].item()
S = st.sidebar.number_input("Stock Price (S0)", value=float(lcallt_price), step=1.0, key="stock price")
K = st.sidebar.number_input("Strike Price (K)", value=float(S * 1.1), step=1.0, key="strike price")
T = st.sidebar.number_input("Time to Maturity (T in years)", value=0.5, key="time")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, key="risk")
N = st.sidebar.number_input("Time Steps (N)", value=250, key="time steps")
M = st.sidebar.number_input("Simulations (M)", value=10000, key="simula")

dt = T / N

if option_type == 'European':
 
 st.subheader('European Option Prices')
    #monte carlo method for european options 
 @st.cache_data
 def european_call_option(S, K, vol, r, N, M, T,Z = None):
        dt = T / N 
        drift = (r - 0.5 * vol**2) * dt
        random_term = vol * np.sqrt(dt)
        lnS = np.log(S) 

        if Z is None: 
           Z = np.random.normal(size=(N,M)) 
            
        delta_lnSt = drift + random_term*Z
        A = np.full((1, M), fill_value=lnS) 
        lnSt = np.concatenate((A,delta_lnSt),axis=0)
        lnst = np.cumsum(lnSt,axis=0)

        St = np.exp(lnst[-1])
        payoff_call = np.maximum(St - K, 0)
        mean_call= np.sum(payoff_call)/M

        #value of the call option 
        call_option = np.exp(-r*T)*mean_call    
        return call_option
 @st.cache_data       
 def european_put_option(S, K, vol, r, N, M, T,Z=None):
        dt = T / N 
        drift = (r - 0.5 * vol**2) * dt
        random_term = vol * np.sqrt(dt)
        lnS = np.log(S) 
        
        #vectorized way 
        if Z is None: 
           Z = np.random.normal(size=(N,M))
        dc = drift + random_term*Z
        A = np.full((1, M), fill_value=lnS)  
        lnSt = np.concatenate((A,dc),axis=0)
        lnst = np.cumsum(lnSt,axis=0)

        St = np.exp(lnst[-1])
        payoff_put = np.maximum(K-St, 0)
        mean_put = np.sum(payoff_put)/M 

        #value of the put option 
        put_option = np.exp(-r*T)*mean_put
        return put_option

    #black scholes model for european options
 @st.cache_data
 def black_scholes(S,K,vol,r,T):
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        call_option = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_option = K * np.exp(-r * T) * norm.cdf(-d2) -  S * norm.cdf(-d1)
        return call_option,put_option

 @st.cache_data   #computing greeks 
 def compute_greeks_formula(S,K,vol,r,T):
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
    #ANALYTICAL GREEKS
        delta_call  = norm.cdf(d1)
        gamma_call = norm.pdf(d1)/(S*vol*np.sqrt(T))
        vega_call = S*norm.pdf(d1)*np.sqrt(T) 
        theta_call = -(S*norm.pdf(d1)*vol)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
        rho_call = K*T*np.exp(-r*T)*norm.cdf(d2)
        delta_put = norm.cdf(d1) - 1
        gamma_put = norm.pdf(d1) / (S * vol * np.sqrt(T))
        vega_put = S * norm.pdf(d1) * np.sqrt(T)
        theta_put = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        

        return delta_call, delta_put, gamma_call, gamma_put, vega_call, vega_put, theta_call, theta_put, rho_call, rho_put

 S_range = np.linspace(50, 150, 40)  # Stock price range

 @st.cache_data
 def compute_greeks_montecarlo(S, K, vol, r, N, M, T):
        E=0.10
        Z = np.random.normal(size=(N, M))

        bcalle = european_call_option(S, K, vol, r, N, M, T,Z)
        price_up = european_call_option(S + E, K, vol, r, N, M, T,Z)
        price_down = european_call_option(S - E, K, vol, r, N, M, T,Z)
        price_vol_up = european_call_option(S, K, vol+E, r, N, M, T,Z)
        price_vol_down= european_call_option(S, K,vol-E, r, N, M, T,Z)
        price_t_up = european_call_option(S, K, vol, r, N, M, T+E,Z)
        price_t_down = european_call_option(S, K, vol, r, N, M, T-E,Z)
        price_r_up = european_call_option(S, K, vol, r+E, N, M, T,Z)
        price_r_down = european_call_option(S, K, vol, r-E, N, M, T,Z)
        bpute = european_call_option(S, K, vol, r, N, M, T,Z)

        price_up_put = european_put_option(S + E, K, vol, r, N, M, T,Z)
        price_down_put = european_put_option(S - E, K, vol, r, N, M, T,Z)
        price_vol_up_put = european_put_option(S, K, vol+E, r, N, M, T,Z)
        price_vol_down_put = european_put_option(S, K,vol-E, r, N, M, T,Z)
        price_t_up_put = european_put_option(S, K, vol, r, N, M, T+E,Z)
        price_t_down_put = european_put_option(S, K, vol, r, N, M, T-E,Z)
        price_r_up_put = european_put_option(S, K, vol, r+E, N, M, T,Z)
        price_r_down_put = european_put_option(S, K, vol, r-E, N, M, T,Z)

        delta = (price_up - price_down) / (2 * E)
        gamma = (price_up - 2 * bcalle + price_down) / (E ** 2)
        vega = (price_vol_up - price_vol_down)/(2 * E)
        theta = (price_t_down -bcalle)/(2 * E)
        rho = (price_r_up - price_r_down)/(2 * E)

        delta_put = (price_up_put - price_down_put) / (2 * E)
        gamma_put = (price_up_put - 2 * bpute + price_down_put) / (E ** 2)
        vega_put = (price_vol_up_put - price_vol_down_put)/(2 * E)
        theta_put = (price_t_down_put - bpute)/(2 * E)
        rho_put = (price_r_up_put - price_r_down_put)/(2 * E)

        return delta, gamma, vega, theta, rho, delta_put, gamma_put, vega_put, theta_put, rho_put

 @st.cache_data   # Monte Carlo pricing function for European Call
 def monte_carlo_price(S0, K, T, r, vol, M):
        Z = np.random.normal(0, 1, M)
        ST = S0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
        payoff = np.maximum(ST - K, 0)  # European Call
        payoff_put = np.maximum(K-ST, 0)
        return np.exp(-r * T) * np.mean(payoff),np.exp(-r * T) * np.mean(payoff_put)

 @st.cache_data  # Estimate Greeks using finite differences
 def estimate_greeks_mc(S0, K, T, r, vol, M, h=1.0, ds=1.0, dr=0.0001, dt=0.01):
        price,price_put = monte_carlo_price(S, K, T, r, vol, M)

        price_up, price_up_put= monte_carlo_price(S + ds, K, T, r, vol, M)
        price_down, price_down_put = monte_carlo_price(S - ds, K, T, r, vol, M)
        delta_call = (price_up - price_down) / (2 * ds)
        gamma_call = (price_up - 2 * price + price_down) / (ds ** 2)
        delta_put = (price_up_put - price_down_put) / (2 * ds)
        gamma_put = (price_up_put - 2 * price_put + price_down_put) / (ds ** 2)

        price_vega,price_vega_put = monte_carlo_price(S, K, T, r, vol + h, M)
        vega_call = (price_vega - price) / h
        vega_put = (price_vega_put - price_put ) / h

        if T - dt > 0:
            price_theta,price_theta_put = monte_carlo_price(S, K, T - dt, r, vol, M)
            theta_call = (price_theta - price) / dt
            theta_put = (price_theta_put - price_put) / dr
        else:
            theta_call= np.nan

        price_rho,price_rho_put = monte_carlo_price(S, K, T, r + dr, vol, M)
        rho_call = (price_rho - price) / dr
        rho_put = (price_rho_put - price_put) / dr
        return delta_call, delta_put, gamma_call, gamma_put, vega_call, vega_put, theta_call, theta_put, rho_call, rho_put

 tab1,tab2,tab3 = st.tabs(['Option Prices','Greeks','Price convergence graphs'])

 n_simulations = 100000 
 drift = (r - 0.5 * vol**2) * T
 random_term = vol * np.sqrt(T)
 Z = np.random.standard_normal(n_simulations)
 ST = S * np.exp((drift+random_term*Z))
 payoffs = np.maximum(ST-K,0) 
 mc_price = np.exp(-r*T)*np.mean(payoffs)

 bs_price , bs_put_price = black_scholes(S, K, vol, r, T)

 with tab3: 
    S = st.number_input("Stock Price (S0)", value=float(lcallt_price), step=1.0,key = 'stock_price')
    K = st.number_input("Strike Price (K)", value=float(S * 1.1), step=1.0,key ='strike_price')
    T = st.number_input("Time to Maturity (T in years)", value=0.5,key='time_to_maturity')
    r = st.number_input("Risk-Free Rate (r)", value=0.05,key ='risk_free_rate')
    N = st.number_input("Time Steps (N)", value=250,key ='time_steps')
    M = st.number_input("Simulations (M)", value=10000,key='simulations')

    st.header('PRICE CONVERGENCE GRAPHS')
    fig, ax = plt.subplots(2,2,figsize=(12,6))

    #plot 1: Terminal price distribution
    ax[0,0].hist(ST,bins = 50, density= True, alpha = 0.7)
    ax[0,0].set_title("Terminal stock price distribution")
    ax[0,0].set_xlabel("stock price")
    ax[0,0].set_ylabel("Density")

    #plot 2 : Payoff distribution 
    ax[0,1].hist(payoffs, bins= 50 , density = True , alpha = 0.7)
    ax[0,1].set_title("Option Payoff distribution")
    ax[0,1].set_xlabel("Payoffs")
    ax[0,1].set_ylabel("Density")

    #plot 3 : convergence analysis 
    n_steps = 10000
    step_size = 100
    n_simulations_list = np.arange(100,100000,step_size)
    mc_prices_call = [] 

    for n in n_simulations_list: 
        Z = np.random.standard_normal(n)
        ST = S*np.exp((drift+random_term*Z))
        payoffs = np.maximum(ST-K,0) 
        mc_prices_call.append(np.exp(-r*T)*np.mean(payoffs))
        
    ax[1,0].semilogx(n_simulations_list, mc_prices_call,'b-' , alpha = 0.7, label = "Monte Carlo")
    ax[1,0].axhline(y = bs_price,color= 'red',linestyle='--', alpha = 0.7,label='Black-Scholes')
    ax[1,0].set_xlabel("number of simulations")
    ax[1,0].set_ylabel("option price")
    ax[1,0].set_title("price convergence of call price option")
    ax[1,0].legend()
    ax[1,0].grid(True)

    mc_prices_put = []
    for n in n_simulations_list: 
        Z = np.random.standard_normal(n)
        ST = S*np.exp((drift+random_term*Z))
        payoffs = np.maximum(K-ST,0) 
        mc_prices_put.append(np.exp(-r*T)*np.mean(payoffs))
        
    ax[1,1].semilogx(n_simulations_list, mc_prices_put,'b-' , alpha = 0.7, label = "Monte Carlo")
    ax[1,1].axhline(y = bs_put_price,color= 'red',linestyle='--', alpha = 0.7,label='Black-Scholes')
    ax[1,1].set_xlabel("number of simulations")
    ax[1,1].set_ylabel("option price")
    ax[1,1].set_title("price convergence of Put price option")
    ax[1,1].legend()
    ax[1,1].grid(True)

    fig.tight_layout()
    st.pyplot(fig)
   
 with tab1: 
   bs_call, bs_put = black_scholes(S, K, vol, r, T)
   call_option = european_call_option(S, K, vol, r, N, M, T,None)
   put_option = european_put_option(S, K, vol, r, N, M, T,None)
   delta_call, delta_put, gamma_call, gamma_put, vega_call, vega_put, theta_call, theta_put, rho_call, rho_put= compute_greeks_formula(S,K,vol,r,T)
   delta, deltap , gamma,gammap,  vega, vegap,theta,thetap, rho,rhop = compute_greeks_montecarlo(S, K, vol, r, N, M, T) 
  
   cont = st.container(border=True)
   
   with cont:
        cont.subheader("💰 Option Prices")
        col1, col2= cont.columns(2)
        
        col1.metric("Monte Carlo Call", f"₹{call_option:.2f}")
        col1.metric("Monte Carlo Put", f"₹{put_option:.2f}")
        col2.metric("Black-Scholes Call", f"₹{np.round(bs_call,2)}")
        col2.metric("Black-Scholes Put", f"₹{np.round(bs_put,2)}")
        

   def plot_simulated_paths(S, vol, r, N, M, T, seed=20):
        np.random.seed(seed)
        dt = T / N
        drift = (r - 0.5 * vol ** 2) * dt
        random_term = vol * np.sqrt(dt)
        lnS = np.log(S)

        Z = np.random.normal(size=(N, M))
        delta_lnSt = drift + random_term * Z
        lnSt = np.concatenate((np.full(shape=(1, M), fill_value=lnS), delta_lnSt), axis=0)
        lnst = np.cumsum(lnSt, axis=0)

        St = np.exp(lnst)

        t = np.arange( N + 1)
    
       # Plot first 10 paths
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(S, color='red', linestyle='--', label=f"Stock ₹{S:.2f}")
        ax.axhline(K, color='green', linestyle='--', label=f"Strike ₹{K:.2f}")
        for i in range(min(200, M)):
           plt.plot(t, St[:, i], lw=1)
        ax.set_title("Simulated GBM Paths for Infosys")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Stock Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        return Z  
  
    # Plot 10 simulated paths
   st.subheader("Simulated Stock Price Paths")
   Z = plot_simulated_paths(S, vol, r, N, M=10000, T=T)

 with tab2: 
   con = st.container(border=True)
   co = st.container(border=True)
   # Streamlit layout
  
   st.title("Monte Carlo Option Greeks Estimation (European Call)")

   with con:      
        con.subheader("Greeks for European Options")
        col4, col5 , col6 , col7,col8 = con.columns(5)
        col4.metric("Delta Call (Analytical) ",f"₹{delta_call:.2f}")
        col4.metric("Delta Call (MC)",f"₹{delta:.2f}")
        col5.metric("Gamma Call (Analytical) ", f"₹{abs(gamma_call) :.2f}")
        col5.metric("Gamma Call (MC)",f"₹{abs(gamma):.2f}")
        col6.metric("Vega Call (Analytical) ",f"₹{vega_call:.2f}")
        col6.metric("vega Call (MC)",f"₹{vega:.2f}")
        col7.metric("theta Call (Analytical) ",f"₹{theta_call:.2f}")
        col7.metric("theta Call (MC)",f"₹{(theta/100):.2f}")
        col8.metric("Rho Call (Analytical)",f"₹{rho_call:.2f}")
        col8.metric("rho Call (MC)",f"₹{rho:.2f}")

   with co:      
        co.subheader("Greeks for European Options")
        col4, col5 , col6 , col7,col8 = co.columns(5)
        col4.metric("Delta Put (Analytical) ",f"₹{delta_put:.2f}")
        col4.metric("Delta Put (MC)",f"₹{deltap:.2f}")
        col5.metric("Gamma Put (Analytical) ", f"₹{abs(gamma_put):.2f}")
        col5.metric("Gamma Put (MC)",f"₹{abs(gammap):.2f}")
        col6.metric("Vega Put (Analytical) ",f"₹{vega_put:.2f}")
        col6.metric("vega Put (MC)",f"₹{vegap:.2f}")
        col7.metric("theta Put (Analytical) ",f"₹{theta_put:.2f}")
        col7.metric("theta Put (MC)",f"₹{thetap:.2f}")
        col8.metric("Rho Put (Analytical)",f"₹{rho_put:.2f}")
        col8.metric("rho Put (MC)",f"₹{rhop:.2f}")

   col1, col2 = st.columns([2, 1])

   with col2:
      st.header("Input Parameters")
      K = st.number_input("Strike Price (K)", value=100.0)
      T = st.number_input("Time to Maturity (T in years)", value=1.0)
      r = st.number_input("Risk-Free Rate (r)", value=0.05)
      vol = st.number_input("Volatility (vol)", value=0.2)
      M = st.number_input("Number of Simulations", value=10000, step=1000)
      S_min = st.number_input("Min Stock Price", value=50.0)
      S_max = st.number_input("Max Stock Price", value=150.0)
      S_step = st.number_input("Step Size", value=5.0)

 # Compute Greeks over range of stock prices
   S_range = np.arange(S_min, S_max + S_step, S_step)
   greeks = {'delta_call':[], 'delta_put':[], 'gamma_call':[], 'gamma_put':[], 'vega_call':[], 'vega_put':[], 'theta_call':[], 'theta_put':[], 'rho_call':[], 'rho_put':[]}

   for S in S_range:
      delta, gamma, vega, theta, rho,dp,gp,vp,tp,rp = estimate_greeks_mc(S, K, T, r, vol, int(M))
      greeks['delta_call'].append(delta)
      greeks['delta_put'].append(gamma)
      greeks['gamma_call'].append(vega)
      greeks['gamma_put'].append(theta)
      greeks['vega_call'].append(rho)
      greeks['vega_put'].append(dp)
      greeks['theta_call'].append(gp)
      greeks['theta_put'].append(vp)
      greeks['rho_call'].append(tp)
      greeks['rho_put'].append(rp)

    # Plotting
   with col1:
        st.header("Greeks vs Stock Price")
        fig, axes = plt.subplots(5, 2, figsize=(14, 10))

        axes[0, 0].plot(S_range, greeks['delta_call'], color='blue')
        axes[0, 0].set_title("Delta Call vs Stock Price")

        axes[0, 1].plot(S_range, greeks['delta_put'], color='green')
        axes[0, 1].set_title("Delta put vs Stock Price")

        axes[1, 0].plot(S_range, greeks['gamma_call'], color='purple')
        axes[1, 0].set_title("Gamma call vs Stock Price")

        axes[1, 1].plot(S_range, greeks['gamma_put'], color='red')
        axes[1, 1].set_title("Gamma put vs Stock Price")

        axes[2, 0].plot(S_range, greeks['vega_call'], color='orange')
        axes[2, 0].set_title("Vega call vs Stock Price")

        axes[2,1].plot(S_range, greeks['vega_put'], color='blue')
        axes[2,1].set_title("Vega put vs Stock Price")

        axes[3,0].plot(S_range, greeks['theta_call'], color='green')
        axes[3,0].set_title("theta call vs Stock Price")

        axes[3,1].plot(S_range, greeks['theta_put'], color='purple')
        axes[3,1].set_title("Theta put vs Stock Price")

        axes[4,0].plot(S_range, greeks['rho_call'], color='red')
        axes[4,0].set_title("Rho Call vs Stock Price")

        axes[4,1].plot(S_range, greeks['rho_put'], color='orange')
        axes[4,1].set_title("Rho put vs Stock Price")

        for ax in axes.flatten():
            ax.set_xlabel("Stock Price")
            ax.set_ylabel("Greek Value")
            ax.grid(True)

        fig.tight_layout()
        st.pyplot(fig)

if option_type == "Asian": 
 st.subheader("Asian Option Prices")
 @st.cache_data
 def average_price_Asian_option(S,K,vol,r,N,M,T):
    dt = T/N 
    drift = (r-(vol**2)*0.5)*dt
    random_term = vol*np.sqrt(dt)
    lns = np.log(S)

#monte carlo method 
    z = np.random.normal(size=(N,M))
    delta_lnst = drift + random_term*z
    a =  np.full(shape=(1,M),fill_value=lns)
    lnSt = np.concatenate((a,delta_lnst),axis = 0 )
    lnst = np.cumsum(lnSt,axis = 0) 
    
#calculating average price    
    st = np.exp(lnst)
    s_avg = np.mean(st[1:],axis=0)
    
#calculating average payoffs for call option
    payoff_call =np.maximum(s_avg - K,0)
    mean_payoff_call = np.mean(payoff_call)
#calculating average payoffs for put option
    payoff_put =np.maximum(K - s_avg ,0)
    mean_payoff_put = np.mean(payoff_put)
#price of the options   
    call_price = np.exp(-r*T)*mean_payoff_call
    put_price = np.exp(-r*T)*mean_payoff_put
         
    return call_price,put_price

 @st.cache_data
# arthimetic average strike asian option
 def average_strike_Asian_option(S,K,vol,r,N,M,T):
    dt = T/N 
    drift = (r-(vol**2)*0.5)*dt
    random_term = vol*np.sqrt(dt)
    lns = np.log(S)

#monte carlo method 
    z = np.random.normal(size=(N,M))
    delta_lnst = drift + random_term*z
    a =  np.full(shape=(1,M),fill_value=lns)
    lnSt = np.concatenate((a,delta_lnst),axis = 0 )
    lnst = np.cumsum(lnSt,axis = 0) 
    
#calculating average price    
    st = np.exp(lnst)
    s_avg = np.mean(st[1:],axis=0)
    St = st[-1]
    
    
#calculating average payoffs for call option
    payoff_call =np.maximum(St - s_avg ,0)
    mean_payoff_call = np.mean(payoff_call)
#calculating average payoffs for put option
    payoff_put =np.maximum(s_avg - St,0)
    mean_payoff_put = np.mean(payoff_put)
#price of the options   
    call_price = np.exp(-r*T)*mean_payoff_call
    put_price = np.exp(-r*T)*mean_payoff_put
         
    return call_price,put_price
 @st.cache_data
 def d1(S, K, T, r, vol):
    return (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
 @st.cache_data
 def d2(S, K, T, r, vol):
    return d1(S, K, T, r, vol) - vol * np.sqrt(T)

 def delta(S, K, T, r, vol, option='call'):
    d_1 = d1(S, K, T, r, vol)
    return norm.cdf(d_1) if option == 'call' else norm.cdf(d_1) - 1

 def gamma(S, K, T, r, vol):
    d_1 = d1(S, K, T, r, vol)
    return norm.pdf(d_1) / (S * vol * np.sqrt(T))

 def theta(S, K, T, r, vol, option='call'):
    d_1 = d1(S, K, T, r, vol)
    d_2 = d2(S, K, T, r, vol)
    term1 = -S * norm.pdf(d_1) * vol / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T)
    if option == 'call':
        return term1 - term2 * norm.cdf(d_2)
    else:
        return term1 + term2 * norm.cdf(-d_2)

 def vega(S, K, T, r, vol):
    d_1 = d1(S, K, T, r, vol)
    return S * norm.pdf(d_1) * np.sqrt(T)

 def rho(S, K, T, r, vol, option='call'):
    d_2 = d2(S, K, T, r, vol)
    if option == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d_2)
    
 def delta_put_asian(S, K, T, r, vol):
    return norm.cdf(d1(S, K, T, r, vol)) - 1

 def gamma_put_asian(S, K, T, r, vol):
    return norm.pdf(d1(S, K, T, r, vol)) / (S * vol * np.sqrt(T))

 def theta_put_asian(S, K, T, r, vol):
    d_1 = d1(S, K, T, r, vol)
    d_2 = d2(S, K, T, r, vol)
    term1 = -S * norm.pdf(d_1) * vol / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d_2)
    return term1 + term2

 def vega_put_asian(S, K, T, r, vol):
    return S * norm.pdf(d1(S, K, T, r, vol)) * np.sqrt(T)

 def rho_put_asian(S, K, T, r, vol):
    return -K * T * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, vol))

 delta_call = delta(S, K, T, r, vol, option='call')
 delta_put = delta_put_asian(S, K, T, r, vol) 
 gamma_call = gamma(S, K, T, r, vol)
 gamma_put = gamma_put_asian(S, K, T, r, vol)
 vega_call= theta(S, K, T, r, vol, option='call')
 vega_put = vega_put_asian(S, K, T, r, vol)
 theta_call = vega(S, K, T, r, vol)
 theta_put = theta_put_asian(S, K, T, r, vol)
 rho_call = rho(S, K, T, r, vol, option='call')
 rho_put = rho_put_asian(S, K, T, r, vol)

 tab1 , tab2,tab3 = st.tabs(["Option Prices",'Greeks','Price Convergence Graphs'])
 with tab1:
  bb1 = st.container(border=True)  
  with bb1: 
        price_call_option, price_put_option  = average_price_Asian_option(S,K,vol,r,N,M,T)
        strike_call_option, strike_put_option = average_strike_Asian_option(S,K,vol,r,N,M,T)

        col1, col2 = st.columns(2)
        col1.metric("Asian Call option (average price)", f"₹{price_call_option:.2f}")
        col1.metric("Asian put option (average price)", f"₹{price_put_option:.2f}")
        col2.metric("Asian Call option (average strike)", f"₹{strike_call_option:.2f}")
        col2.metric("Asian put option (average strike)", f"₹{strike_put_option:.2f}")
    
  bb2 = st.container(border=True) 
  with bb2:
     def plot_simulated_paths(S, vol, r, N, M, T, seed=20):
        np.random.seed(seed)
        dt = T / N
        drift = (r - 0.5 * vol ** 2) * dt
        random_term = vol * np.sqrt(dt)
        lnS = np.log(S)

        Z = np.random.normal(size=(N, M))
        delta_lnSt = drift + random_term * Z
        lnSt = np.concatenate((np.full(shape=(1, M), fill_value=lnS), delta_lnSt), axis=0)
        lnst = np.cumsum(lnSt, axis=0)

        St = np.exp(lnst)

        t = np.arange( N + 1)
    
       # Plot first 10 paths
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(S, color='red', linestyle='--', label=f"Stock ₹{S:.2f}")
        ax.axhline(K, color='green', linestyle='--', label=f"Strike ₹{K:.2f}")
        for i in range(min(200, M)):
           plt.plot(t, St[:, i], lw=1)
        ax.set_title("Simulated GBM Paths for Infosys")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Stock Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        return Z  
      # Plot 10 simulated paths


  st.subheader("Simulated Stock Price Paths")
  Z = plot_simulated_paths(S, vol, r, N, M=10000, T=T)
     
 with tab2:   
  bb3 = st.container(border=True)
  bb4 = st.container(border=True)
  with bb3: 
        delta_call = delta(S, K, T, r, vol, option='call')
        delta_put = delta_put_asian(S, K, T, r, vol) 
        gamma_call = gamma(S, K, T, r, vol)
        gamma_put = gamma_put_asian(S, K, T, r, vol)
        vega_call=  vega(S, K, T, r, vol)
        vega_put = vega_put_asian(S, K, T, r, vol)
        theta_call = theta(S, K, T, r, vol, option='call')
        theta_put = theta_put_asian(S, K, T, r, vol)
        rho_call = rho(S, K, T, r, vol, option='call')
        rho_put = rho_put_asian(S, K, T, r, vol)
        row1,row2,row3,row4,row5 = bb3.columns(5)
        
        row1.metric("Delta Call",f"₹{delta_call:.2f}")
        row1.metric("Delta Put",f"₹{delta_put:.2f}")
        row2.metric("Gamma Call ", f"₹{gamma_call:.2f}")
        row2.metric("Gamma put",f"₹{gamma_put:.2f}")
        row3.metric("Vega Call ",f"₹{vega_call:.2f}")
        row3.metric("vega put",f"₹{vega_put:.2f}")
        row4.metric("theta Call ",f"₹{theta_call:.2f}")
        row4.metric("theta put",f"₹{theta_put:.2f}")
        row5.metric("Rho Call ",f"₹{rho_call:.2f}")
        row5.metric("rho put",f"₹{rho_put:.2f}")


  S_range = np.linspace(S * 0.8, S * 1.2, 50)
  delta_list = [delta(s, K, T, r, vol) for s in S_range]
  gamma_list = [gamma(s, K, T, r, vol) for s in S_range]
  theta_list = [theta(s, K, T, r, vol) for s in S_range]
  vega_list =  [vega(s, K, T, r, vol) for s in S_range]
  rho_list =   [rho(s, K, T, r, vol) for s in S_range]

  with bb4: 
        bb4.header("Greeks vs Stock Price")
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))

        axes[0, 0].plot(S_range, delta_list, color='blue')
        axes[0, 0].set_title("Delta vs Stock Price")

        axes[0, 1].plot(S_range, gamma_list, color='green')
        axes[0, 1].set_title("Gamma vs Stock Price")

        axes[1, 0].plot(S_range, vega_list, color='purple')
        axes[1, 0].set_title("Vega vs Stock Price")

        axes[1, 1].plot(S_range, theta_list, color='red')
        axes[1, 1].set_title("Theta vs Stock Price")

        axes[2, 0].plot(S_range, rho_list, color='orange')
        axes[2, 0].set_title("Rho vs Stock Price")

        for ax in axes.flatten():
                ax.set_xlabel("Stock Price")
                ax.set_ylabel("Greek Value")
                ax.grid(True)

  fig.tight_layout()
  st.pyplot(fig)

 with tab3:

    fig, axi = plt.subplots(2, 2, figsize=(12, 6))
    n_simulations_list = np.arange(100, 10001, 500)  # reduced max

    S = st.number_input("Spot Price (S0)", value=float(lcallt_price), step=1.0)
    K = st.number_input("Strike Price (K)", value=float(S * 1.1), step=1.0)
    T = st.number_input("Time to Maturity (T in years)", value=0.5)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    N = st.number_input("Time Steps (N)", value=250)
    M = st.number_input("Simulations (M)", value=1000)
     
    def simulate_payoffs(sim_type):
        prices = []
        for n in n_simulations_list:
            z = np.random.normal(size=(N, n))
            drift = (r-(vol**2)*0.5)*dt
            lnS = np.log(S) + np.cumsum((drift * dt + vol * np.sqrt(dt) * z), axis=0)
            sT = np.exp(np.vstack([np.full((1, n), np.log(S)), lnS]))
            s_avg = np.mean(sT[1:], axis=0)
            ST_final = sT[-1]

            if sim_type == "asian_call":
                payoff = np.maximum(s_avg - K, 0)
            elif sim_type == "asian_put":
                payoff = np.maximum(K - s_avg, 0)
            elif sim_type == "avg_strike_call":
                payoff = np.maximum(ST_final - s_avg, 0)
            else:  # avg_strike_put
                payoff = np.maximum(s_avg - ST_final, 0)

            prices.append(np.exp(-r * T) * np.mean(payoff))
        return prices

    types = ["asian_call", "asian_put", "avg_strike_call", "avg_strike_put"]
    titles = ["Asian Call", "Asian Put", "Avg Strike Call", "Avg Strike Put"]

    for i, option_type in enumerate(types):
        row, col = divmod(i, 2)
        prices = simulate_payoffs(option_type)
        axi[row, col].semilogx(n_simulations_list, prices, label=titles[i])
        axi[row, col].set_title(titles[i])
        axi[row, col].set_xlabel("Simulations")
        axi[row, col].set_ylabel("Price")
        axi[row, col].legend()
        axi[row, col].grid(True)

    box = st.container(border=True)
    with box: 
        fig.tight_layout()
        st.pyplot(fig)
    
    
