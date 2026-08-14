# AI Notes - Better Finance Project B

## How I used AI

I used ChatGPT as a coding and debugging assistant during Project B. The main uses were to help translate the assignment requirements into a working design, draft and review portfolio/sentiment/fusion code, check the Streamlit workflow, diagnose runtime errors, and organise the evidence needed for the report and AI workflow record.

I did not treat the first AI response as final. I compared suggestions with the assignment brief, the inherited Project A code, the generated output files, and actual runtime behaviour. Several important parts were changed after that review.

## Important corrections I made to AI output

### 1. Product continuity

An early AI design renamed the product. I rejected that change because the product had already been defined as **Better Finance**. I changed the Project B interface, metadata, and descriptions so they continue the same product identity and target-user idea.

### 2. Scope of the fund offering

I did not use a large collection of highly complex funds. The final implementation has three base combined funds—Equal Weight, Minimum Variance, and Risk Parity—plus a Minimum Variance fund with a sentiment tilt. This is above the assignment minimum but still consistent with the simple Better Finance value proposition.

### 3. Preserving inherited code

I required the inherited `etl.py` and `features.py` to remain unchanged. I also kept the starter public function interfaces in the Project B model files and used underscore-prefixed helper functions for additional logic. This reduced the risk of creating a second incompatible pipeline.

### 4. Avoiding incomplete sample data

The earlier work included complete stock and crypto return outputs but only sample versions of some other displayed data. I did not use those sample files as the full modelling dataset. Complete returns were reused directly, while the full headline panel was recreated through the inherited loading and feature functions.

### 5. Removing hard-coded paths

An earlier AI draft embedded a specific sibling-project name. I treated that as too rigid even though it was not a laptop absolute path. I changed the logic to resolve locations dynamically from the current project root and allow an environment-variable override.

### 6. Checking look-ahead with output evidence

I did not rely only on reading the walk-forward loop. I checked the generated rebalance audit. There are 108 audited rebalance rows, all have `past_only_check = True`, and every regular estimation window contains 252 observations. The first live date is 4 January 2021, after the initial training period.

### 7. Keeping sentiment equity-only

The supplied news data applies to equities, not crypto. I therefore did not create a synthetic crypto sentiment score. The sentiment extension adjusts only the equity sleeve of the combined Minimum Variance fund and keeps the crypto sleeve budget unchanged.

### 8. Preventing same-day sentiment use

The sector sentiment signal is shifted by one equity trading day before portfolio use. Ticker-days without headlines are treated as neutral, while coverage is recorded separately. This makes the timing rule explicit and avoids using day-t news in a day-t decision.

### 9. Keeping a negative fusion result

The sentiment extension did not improve the final out-of-sample metrics. I did not tune the coefficient after seeing this result. The base Minimum Variance fund has an annualised return of about 5.78%, volatility of about 12.69%, Sharpe of about 0.506, and maximum drawdown of about -14.87%. The sentiment version has an annualised return of about 5.72%, volatility of about 12.69%, Sharpe of about 0.502, and maximum drawdown of about -14.92%. The Sharpe change is about -0.005. I kept this result because the assignment explicitly allows an honest negative fusion result.

### 10. Separating build-time and app-time work

The full script performs the heavy work and writes precomputed results. The Streamlit app only reads those results. It does not import NLTK, score headlines, or rerun portfolio optimisation when a user opens the app. This was checked because recomputing models in the deployed interface would make the app less reliable.

### 11. Debugging the environment before editing source code

Several local failures came from the Python environment rather than the project logic. I checked package versions and compiled extensions, repaired the affected packages, and installed the VADER lexicon before changing source code. After the environment was repaired, the full reproduction script ran successfully and the Streamlit app loaded locally without an application error.

## What the final results say

The four investable variants cover three base methods and one news-sentiment extension. Over the 2021-2023 out-of-sample period:

- **Combined Equal Weight** has the highest annualised return at about 14.98%, with volatility of about 21.25%, Sharpe of about 0.763, and maximum drawdown of about -28.75%.
- **Combined Risk Parity** has the strongest risk-adjusted result, with annualised return of about 14.01%, volatility of about 16.01%, Sharpe of about 0.899, and maximum drawdown of about -19.71%.
- **Combined Minimum Variance** has the lowest volatility at about 12.69%, with annualised return of about 5.78%, Sharpe of about 0.506, and maximum drawdown of about -14.87%.
- **Combined Minimum Variance + Sentiment** is very close to the base Minimum Variance fund but slightly weaker on return, Sharpe, and drawdown.

The sector sentiment output covers 10 equity sectors from January 2020 to December 2023. Average ticker coverage is about 75%, with a median of 80%. This means the standalone sentiment index is usable as an analytic, but the simple lagged tilt did not add meaningful out-of-sample value to the Minimum Variance fund.

## 

## Final verification approach

Before submission I would rerun the full build, confirm the required outputs exist with the exact filenames, run the hand-in checker, open every app tab locally, verify the public deployment in a logged-out browser session, and review the report so that every required figure/table is interpreted in my own words.
