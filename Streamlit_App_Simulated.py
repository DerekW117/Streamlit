

# Load model and data files
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "market_model.pkl")
sp500_path = os.path.join(script_dir, "aapl_sp500_volatility.csv")
fed_path = os.path.join(script_dir, "fed_funds_rate.csv")
sentiment_path = os.path.join(script_dir, "news_sentiment_data.xlsx")

model = joblib.load(model_path)
data = pd.read_csv(sp500_path)
fed_data = pd.read_csv(fed_path)
sentiment_data = pd.read_excel(sentiment_path)

# Fetch AAPL historical data from Yahoo Finance
aapl_data = yf.download("AAPL", period="5y")
aapl_data.reset_index(inplace=True)
aapl_data = aapl_data.sort_values('Date')
aapl_data['7d_avg'] = aapl_data['Close'].shift(-1).rolling(window=7).mean()

st.title("📈 AAPL Stock Price Predictor (Next 7-Day Average)")
st.markdown("This app lets you input macro and sentiment factors to estimate Apple's average stock price over the next week.")

# User Inputs
fed_rate = st.slider("Current Federal Funds Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25)
sentiment = st.selectbox("Is economic news sentiment optimistic?", options=["Yes", "No"])
sentiment_val = 1 if sentiment == "Yes" else 0

volatility = st.slider("How volatile is the market? (1 = Low, 10 = High)", 1, 10, 5)
volatility_score = volatility / 10

volume_flag = st.selectbox("Is AAPL trading volume high recently?", options=["Yes", "No"])
volume_val = 1 if volume_flag == "Yes" else 0

sp500_trend = st.selectbox("Has the S&P 500 been rising or falling?", options=["Rising", "Falling"])
sp500_return = 1 if sp500_trend == "Rising" else 0

# Feature definition reminder
st.markdown("### Feature Summary")
st.code(\"\"\"
FEDFUNDS: Federal funds interest rate
sentiment: 1 = optimistic news, 0 = pessimistic
AAPL_21d_Volatility: market volatility, normalized 0–1
AAPL_HighVolume: 1 = high trading volume, 0 = low
SP500_Return: 1 = S&P500 rising, 0 = falling
\"\"\")

# Construct input DataFrame
input_df = pd.DataFrame({
    'FEDFUNDS': [fed_rate],
    'sentiment': [sentiment_val],
    'AAPL_21d_Volatility': [volatility_score],
    'AAPL_HighVolume': [volume_val],
    'SP500_Return': [sp500_return]
})

st.markdown("---")
st.subheader("🔍 Model Input Preview")
st.write(input_df)
st.markdown("---")

# Predict
if st.button("Predict AAPL 7-Day Average Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted 7-day average AAPL price: ${prediction:.2f}")
"""
