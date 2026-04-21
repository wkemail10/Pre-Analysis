# Ticker Advisor

A deployment-ready Streamlit app that takes only a ticker and returns:
- Buy
- Sell / Avoid
- No Action / Watch

It automatically checks:
- trend
- RSI
- MACD
- volume
- simple candle/pattern signals
- SPY / QQQ / DXY / 10Y trend
- recent headlines
- optional analyst and insider data when a Finnhub key is added

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deploy

1. Upload these files to a GitHub repo.
2. Go to Streamlit Community Cloud.
3. Create app.
4. Choose your repo, branch `main`, file `app.py`.
5. Deploy.

## Optional secrets

If you want richer analyst / insider / company-news data, add this secret:

```toml
FINNHUB_API_KEY = "your_key_here"
```

On Streamlit Cloud, add it in the app Secrets panel.

## Notes

- Yahoo Finance data is used by default.
- Finnhub is optional.
- This is a decision-support tool, not guaranteed financial advice.
