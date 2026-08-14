# ChatGPT.md - Project Instructions for Better Finance

## Purpose

This file records the working instructions used with ChatGPT for the Better Finance investment-app project. The goal is to extend the existing product concept into systematic multi-asset funds, a news-sentiment analytic, and a lightweight Streamlit interface while preserving the student's own earlier data foundation.

## Product continuity

- Keep the product name **Better Finance** unchanged.
- Preserve the existing target-user idea: busy users with limited time or technical financial knowledge who benefit from simple comparisons and clear explanations.
- Keep the interface understandable rather than turning the product into a professional trading terminal.
- New features should support the original value proposition rather than replace it.

## Data and code continuity

- Treat the existing `etl.py` and `features.py` as fixed inherited work. Do not modify their contents.
- Reuse complete derived return outputs whenever they are available.
- Do not use display-only sample files as full modelling inputs.
- When a complete news panel is not stored, rebuild it only through the inherited cleaning and feature functions rather than creating a second cleaning pipeline.
- Load raw data only through the provided data-access helper.
- Do not hard-code laptop-specific or student-specific absolute paths. Resolve locations relative to the current project root or through an environment variable.

## Starter-interface rules

- Keep the provided public interfaces in `portfolios.py`, `sentiment.py`, and `fusion.py`.
- Do not add new public functions to those files.
- Additional helpers must begin with an underscore.
- Do not create extra helper Python modules for Project B logic.
- Keep the main runner as an orchestration layer rather than duplicating model logic there.

## Portfolio rules

- The minimum investable product must combine equities and crypto.
- Provide at least two portfolio methods; the implemented set uses Equal Weight, Minimum Variance, and Risk Parity.
- Use a 252-observation estimation window for the combined fund and rebalance every 21 equity trading days.
- The first live return must occur only after the initial estimation window.
- At every rebalance, weights must be estimated from information dated strictly before the rebalance date.
- Keep portfolios long-only and fully invested.
- Limit total crypto exposure to 25% for optimised combined funds.
- Use the equity trading calendar for the combined fund.
- State the zero risk-free-rate assumption used for Sharpe.
- Zero transaction costs are acceptable for this version but must be stated explicitly.
- Save a rebalance audit that makes the no-look-ahead rule directly checkable.

## Performance and fact-sheet rules

For every fund, calculate and expose at least:

- growth of $1;
- annualised return;
- annualised volatility;
- Sharpe ratio;
- maximum drawdown;
- latest target holdings.

Use the exact required output filenames from the assignment brief. Additional outputs may be added when they directly support the report or app.

## Sentiment rules

- Use VADER on the assembled equity headlines.
- Preserve the text in a form suitable for VADER; do not strip information merely for convenience.
- Aggregate headline scores to ticker-day first, then equal-weight ticker-day sentiment within each equity sector.
- Treat ticker-days without headlines as neutral and record coverage separately.
- Crypto is price-only and must not be assigned an invented headline sentiment score.
- Lag the sector sentiment signal by at least one real equity trading day before it is used in a portfolio decision.

## Fusion rules

- Use the lagged equity-sector sentiment as a simple tilt to a base Minimum Variance combined fund.
- Apply the sentiment adjustment only to the equity sleeve.
- Preserve the base crypto sleeve budget because no crypto headline signal is available.
- Re-normalise the adjusted equity sleeve so the total portfolio remains fully invested.
- Compare the base and sentiment-augmented funds using the same out-of-sample period.
- Do not tune the tilt after inspecting the final out-of-sample result merely to manufacture outperformance.
- Keep and explain a negative result if the sentiment extension does not improve performance.

## Streamlit rules

- The deployed app must read precomputed results rather than rerun optimisation, VADER, or the full backtest.
- The deployed app must not import NLTK.
- Keep the investor journey complete: compare funds, inspect a fact sheet, set an allocation, and explore sector sentiment.
- Keep the design simple and visually coherent with Better Finance.
- Use concise language suitable for non-technical users.
- Show the sentiment-fusion comparison rather than hiding an unfavourable result.

## Validation rules

Before accepting AI-generated changes:

1. compile the edited Python files;
2. run the full reproduction script;
3. check that every rebalance audit row passes the past-only test;
4. confirm the required output files exist with the exact names;
5. inspect the fund metrics for implausible values;
6. compare the base and sentiment-augmented fund using actual results;
7. run the Streamlit app locally and click through every tab;
8. run the provided hand-in checker;
9. remove caches, secrets, raw data, and local-environment files before submission.

## How ChatGPT output should be treated

ChatGPT is used for drafting code, debugging, interface ideas, and documentation structure. Its output is not accepted automatically. Changes must be checked against the assignment brief, the inherited code, actual output files, and local runtime behaviour. When an AI suggestion changes an assumption, product name, data source, path, public API, or timing convention, verify it explicitly before keeping it.
