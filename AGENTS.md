# AGENTS.md - ChatGPT Workflow Compatibility File

The primary AI assistant used for this project was **ChatGPT**. The detailed working instructions are recorded in `ChatGPT.md`.

The following rules were treated as non-negotiable throughout the build:

- keep the product name **Better Finance** and continue the existing product concept;
- preserve `etl.py` and `features.py` unchanged from the earlier project work;
- retain the starter public interfaces and use underscore-prefixed helpers for extensions;
- do not create additional helper Python modules for the Project B model logic;
- avoid hard-coded machine paths;
- form out-of-sample weights from past information only;
- use equity news sentiment only for equity assets and lag it before portfolio use;
- keep the deployed app lightweight by reading precomputed outputs;
- verify AI-produced code with runtime tests, output checks, and the assignment brief before accepting it.

`ChatGPT.md` contains the full project-specific instructions and validation workflow.
