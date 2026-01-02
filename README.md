# Stock Analysis System (Paper Trading)

This repo runs a daily scheduled GitHub Action to:
- fetch prices for the configured tickers
- fetch recent news headlines (RSS) for each ticker
- compute simple, transparent signals (momentum + risk filters)
- generate daily/weekly/monthly outlook tables
- publish a lightweight dashboard via GitHub Pages

## Quick start (GitHub)
1. Create a new GitHub repo and upload the contents of this folder (or upload the ZIP).
2. In GitHub: **Settings → Pages** → Source: `Deploy from a branch`, Branch: `main` and folder: `/docs`.
3. In GitHub: **Actions** → enable workflows (if prompted).
4. Done. The dashboard will be available at: `https://<username>.github.io/<repo>/`

## Configure
Edit `config/universe.json` to change tickers and categories.

## Notes
- Prices are fetched via Stooq CSV endpoints when available.
- News is fetched via Google News RSS search URLs.
- This is **paper-trading research**. No brokerage integration is included.
