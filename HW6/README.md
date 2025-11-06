# Gold & Cryptocurrency 2022 Sentiment Pipeline

This project calls GPT-4 API twice to generate:
1. A brief report on the return on investment in the **gold** market in 2022.
2. A brief report on the return on investment in the **cryptocurrency** market in 2022.

Then it runs **sentiment analysis** on both GPT responses using a **Longformer**-based sentiment model (or another Longformer checkpoint specified in the config).

Finally, it writes a short comparative report to `reports/sentiment_report.md`.

## Steps

1. Create and activate virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
