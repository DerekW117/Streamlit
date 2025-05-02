
import streamlit as st
import pandas as pd
import os
import xgboost as xgb

# 设置路径
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "xgb_model.json")

# 加载模型
model = None
try:
    model = xgb.XGBRegressor()
    model.load_model(model_path)
except Exception as e:
    st.warning(f"⚠️ 无法加载模型文件：{e}")

# 应用标题和说明
st.title("📈 AAPL Stock Price Predictor (Next 7-Day Average)")
st.markdown("输入宏观和情绪因素，预测未来一周苹果股价平均值。")

# 用户输入
fed_rate = st.slider("当前联邦基金利率（%）", 0.0, 10.0, 5.0, 0.25)
sentiment = st.selectbox("经济新闻情绪是否乐观？", ["Yes", "No"])
sentiment_val = 1 if sentiment == "Yes" else 0
volatility = st.slider("市场波动性（1 = 低，10 = 高）", 1, 10, 5)
volatility_score = volatility / 10
volume_flag = st.selectbox("AAPL 最近交易量高吗？", ["Yes", "No"])
volume_val = 1 if volume_flag == "Yes" else 0
sp500_trend = st.selectbox("标普500 是上涨还是下跌？", ["Rising", "Falling"])
sp500_return = 1 if sp500_trend == "Rising" else 0

# 展示输入特征
st.markdown("### 特征说明")
st.code("""
FEDFUNDS: 联邦基金利率
sentiment: 情绪（1 = 乐观，0 = 悲观）
AAPL_21d_Volatility: 市场波动（归一化）
AAPL_HighVolume: 交易量高（1 = 是）
SP500_Return: 标普上涨（1 = 是）
""")

# 构建输入数据
input_df = pd.DataFrame({
    'FEDFUNDS': [fed_rate],
    'sentiment': [sentiment_val],
    'AAPL_21d_Volatility': [volatility_score],
    'AAPL_HighVolume': [volume_val],
    'SP500_Return': [sp500_return]
})

st.markdown("---")
st.subheader("🔍 模型输入预览")
st.write(input_df)
st.markdown("---")

# 预测
if st.button("预测苹果未来7日平均股价"):
    if model is not None:
        prediction = model.predict(input_df)[0]
        st.success(f"📊 预测苹果未来7日平均股价为：${prediction:.2f}")
    else:
        st.error("❌ 模型未加载，无法进行预测。请联系管理员上传正确模型文件。")
