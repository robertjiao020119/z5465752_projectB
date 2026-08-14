# Prompt log - Better Finance Project B

## Entry 1 - Extending the existing product into Project B

### What I wanted
I wanted to continue the existing Better Finance product into the funds, sentiment, and app stage without changing the original product identity. I wanted the implementation to meet the assignment minimum while adding a small amount of extra functionality.

### Prompt(s)
“Continue my existing Better Finance product into Project B. Reuse my earlier work, keep the product idea consistent, meet the required combined equity-and-crypto fund requirement, and add only a modest extension rather than making the project unnecessarily complex.”

### What the assistant produced
The assistant proposed a combined-fund product with Equal Weight, Minimum Variance, Risk Parity, and a sentiment-augmented Minimum Variance version. It also proposed a simple investor app with fund comparison, fact sheets, allocation, and sentiment views.

### What was wrong or risky
An early draft changed the product name instead of continuing Better Finance. That would have broken continuity with the earlier product design and made the two parts look disconnected.

### What I changed and why
I required the product name to remain **Better Finance** everywhere, including the app, metadata, and explanatory text. I kept the fund set modest: three base methods plus one sentiment extension. This better matches the original product idea and the intended scope.

---

## Entry 2 - Preserving inherited code and starter interfaces

### What I wanted
I wanted the Project B code to extend the starter without changing the inherited cleaning and feature-engineering work or expanding the public API.

### Prompt(s)
“Do not modify my existing `etl.py` or `features.py`. Keep the starter public functions in the portfolio, sentiment, and fusion files. Extra logic can use underscore-prefixed helpers, but do not add new public functions or extra helper Python modules.”

### What the assistant produced
The assistant placed the extra implementation inside private helper functions while keeping `oos_backtest`, `performance_metrics`, `score_headlines`, `sector_sentiment_index`, and `apply_sentiment` as the public interfaces.

### What was wrong or risky
AI-generated code can easily introduce convenience functions as new public APIs or duplicate inherited logic in a new module. That would make the submission diverge from the starter structure and make the code harder to audit.

### What I changed and why
I checked the public function names and required every added helper to start with an underscore. I also checked that the inherited `etl.py` and `features.py` remained unchanged. This kept the code structure close to the starter and made responsibilities clearer.

---

## Entry 3 - Reusing the earlier data foundation correctly

### What I wanted
I wanted Project B to reuse the complete outputs from the earlier work rather than recomputing everything or mistakenly modelling from sample display files.

### Prompt(s)
“Explain which earlier outputs are complete enough to reuse for modelling. Use complete stock and crypto return outputs directly. Do not treat sample combined-return or sample headline files as full modelling data.”

### What the assistant produced
The assistant reused the complete stock and crypto return matrices, aligned them to form the combined return panel, and used the inherited news-loading and headline-assembly functions to recreate the full headline panel when only a sample had been stored previously.

### What was wrong or risky
Using a display sample as the complete modelling dataset would shorten the backtest and distort the sentiment index. Rebuilding returns from raw prices unnecessarily would also duplicate earlier work.

### What I changed and why
I kept the complete return artifacts as the main inputs and used the inherited functions only where a full headline panel was required. This preserves the earlier data foundation while avoiding the use of incomplete sample artifacts.

---

## Entry 4 - Removing hard-coded paths

### What I wanted
I wanted the project to run on another computer and on a deployment platform without editing student-specific or laptop-specific paths.

### Prompt(s)
“Check the complete Project B code for hard-coded paths. The assignment says not to use laptop-specific paths. Replace any student-specific path construction with a portable approach.”

### What the assistant produced
The assistant initially used a fixed sibling project name to locate earlier outputs, then revised the runner to resolve locations relative to the current project root and to allow an environment-variable override.

### What was wrong or risky
Even though the first path was not an absolute `C:\` or `D:\` path, it still wrote a specific project name into the code. That made the reuse logic less portable and was inconsistent with the path requirement.

### What I changed and why
I removed the fixed project name and used dynamic root-relative resolution with an optional environment variable. I also kept all output locations relative to the project root. This lets the same code run without editing paths on another machine.

---

## Entry 5 - Building a no-look-ahead walk-forward backtest

### What I wanted
I wanted a correct out-of-sample backtest with monthly-style rebalancing and weights estimated only from historical observations.

### Prompt(s)
“Build the combined funds with a 252-observation estimation window and rebalance every 21 equity trading days. Make sure the training data ends before the rebalance date and save an audit that proves the past-only rule.”

### What the assistant produced
The assistant implemented Equal Weight, Minimum Variance, and Risk Parity funds. The resulting audit records the training start, training end, rebalance date, holding end, training observations, and a `past_only_check` flag.

### What was wrong or risky
A walk-forward loop can accidentally include the rebalance-day return in the estimation window, which would create look-ahead bias. It can also accidentally annualise a combined fund on a crypto calendar.

### What I changed and why
I checked the generated audit rather than relying only on the code. All 108 rebalance audit rows have `past_only_check = True`, use 252 training observations, and the first live date is 4 January 2021. The combined fund uses the equity trading calendar and 252-day annualisation.

---

## Entry 6 - Sector sentiment and equity-only fusion

### What I wanted
I wanted to meet the sentiment and structured/unstructured fusion requirement without inventing a crypto news signal that does not exist in the supplied data.

### Prompt(s)
“Use VADER to build a sector sentiment index. Treat no-news ticker-days explicitly, lag sentiment by one real trading day, and use it in a simple fund tilt. Crypto has no headline data, so do not assign it an artificial sentiment score.”

### What the assistant produced
The assistant scored equity headlines with VADER, aggregated to ticker-day and then equal-weighted sector sentiment, treated no-news ticker-days as neutral, recorded coverage, and created a one-trading-day lag. The fusion tilts the equity weights of the Minimum Variance combined fund while keeping the crypto sleeve budget unchanged.

### What was wrong or risky
Using same-day sentiment would create look-ahead. Applying equity-sector sentiment directly to crypto would have no data justification. Dropping every no-news ticker-day could also change the meaning of the sector average.

### What I changed and why
I kept sentiment equity-only, used the lagged signal, preserved the crypto sleeve, and kept the neutral no-news rule with a separate coverage measure. This makes the fusion transparent and consistent with the data available.

---

## Entry 7 - Checking the actual fund results instead of tuning to win

### What I wanted
I wanted to assess the model from actual out-of-sample results and keep the sentiment extension even if it did not improve performance.

### Prompt(s)
“Read the generated performance and fusion outputs. Compare the funds using annualised return, volatility, Sharpe, and drawdown. Do not change the sentiment tilt just to force a better final result.”

### What the assistant produced
The results show that Combined Risk Parity has the highest Sharpe ratio, Combined Equal Weight has the highest annualised return, and Combined Minimum Variance has the lowest volatility. The sentiment extension produces a slightly lower Sharpe ratio than the base Minimum Variance fund.

### What was wrong or risky
It would be easy to tune the sentiment coefficient after seeing the final out-of-sample result and then report the best-looking value. That would weaken the credibility of the evaluation.

### What I changed and why
I kept the fixed simple tilt and retained the negative before-versus-after result. The base Minimum Variance Sharpe is approximately 0.506 and the sentiment version is approximately 0.502, a change of about -0.005. I treated this as evidence that the simple signal adds little incremental value in this sample rather than hiding it.

---

## Entry 8 - Designing a lightweight app from precomputed results

### What I wanted
I wanted the app to cover the full investor journey while remaining simple enough to deploy reliably.

### Prompt(s)
“Build a Better Finance Streamlit app that compares funds, shows a fund fact sheet, lets a user set an allocation, and displays sector sentiment. It must load precomputed outputs and must not rerun VADER or portfolio optimisation in the deployed app.”

### What the assistant produced
The assistant created five tabs: fund comparison, fund details, allocation, news mood, and an about/data page. The app reads the saved fund returns, weights, sentiment index, performance metrics, and fusion comparison.

### What was wrong or risky
A deployed app that imports NLTK or reruns the walk-forward optimisation would be slow and may fail on a basic cloud instance. Generic warning text also made the interface feel more like a classroom template than the Better Finance product.

### What I changed and why
I kept all heavy computation in the reproduction script and confirmed that the app does not import NLTK. I also simplified the user-facing wording, removed generic prototype/disclaimer text, and changed the visual palette to a blue-indigo design while keeping the interface easy to understand.

---

## Entry 9 - Debugging the local Python environment without changing model code

### What I wanted
I wanted to distinguish code bugs from local package-installation problems when the full script failed before or during imports.

### Prompt(s)
“Diagnose the traceback before changing Project B code. Check whether NumPy, Matplotlib, regex, and the VADER lexicon are installed correctly for Python 3.12.”

### What the assistant produced
The assistant identified several environment-level failures, including incompatible compiled package files and a missing VADER resource, and suggested package-level checks before editing source code.

### What was wrong or risky
Changing portfolio or sentiment source code in response to a broken binary package would have created unnecessary code changes and could have hidden the real problem.

### What I changed and why
I repaired the local package environment, verified the core scientific packages could import, installed the required VADER resource, then reran the original Project B code. The full reproduction script completed successfully and generated the expected result files, and the Streamlit app then ran locally without an application error.
