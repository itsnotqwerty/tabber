# Tabber

An OSINT CLI tool that determines the most likely current or recent physical location of a public figure by aggregating data from multiple sources and reasoning over them with an LLM.

## How It Works

1. **Disambiguation** — The input name is resolved to a structured person profile via LLM.
2. **Feedback loop** (up to N iterations):
   - The LLM generates targeted search hints based on the profile and any prior data.
   - All configured gatherers run in parallel to collect raw data.
   - The LLM evaluates whether there is sufficient location signal; exits early if so, or refines and repeats.
3. **Location analysis** — All gathered data is synthesised by a final LLM call into a location result with confidence and reasoning.
4. Results are displayed in a Rich terminal panel with colour-coded confidence.

## Installation

```bash
git clone https://github.com/yourname/tabber.git
cd tabber
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Configuration is stored in `~/.tabber/config.json` and managed via the `config` subcommand.

```bash
python tabber.py config set <key> <value>
python tabber.py config show
```

### Configuration Keys

+--------------------------+----------+----------------------------------------------+
| Key                      | Default  | Description                                  |
+==========================+==========+==============================================+
| `max_iterations`         | `3`      | Max feedback loop iterations                 |
+--------------------------+----------+----------------------------------------------+
| `llm_provider`           | `openai` | LLM backend to use (`openai` or `anthropic`) |
+--------------------------+----------+----------------------------------------------+
| `openai_api_key`         | —        | Required when `llm_provider` is `openai`     |
+--------------------------+----------+----------------------------------------------+
| `anthropic_api_key`      | —        | Required when `llm_provider` is `anthropic`  |
+--------------------------+----------+----------------------------------------------+
| `twitter_bearer_token`   | —        | Enables the Twitter gatherer                 |
+--------------------------+----------+----------------------------------------------+
| `instagram_access_token` | —        | Enables the Instagram gatherer               |
+--------------------------+----------+----------------------------------------------+
| `reddit_client_id`       | —        | Required (with secret) for Reddit gatherer   |
+--------------------------+----------+----------------------------------------------+
| `reddit_client_secret`   | —        | Required (with ID) for Reddit gatherer       |
+--------------------------+----------+----------------------------------------------+

At minimum, set your API key for the chosen provider:

```bash
# OpenAI (default)
python tabber.py config set openai_api_key sk-...

# Anthropic
python tabber.py config set llm_provider anthropic
python tabber.py config set anthropic_api_key sk-ant-...
```

## Usage

```bash
python tabber.py lookup "Elon Musk"
python tabber.py "Elon Musk"        # shorthand
```

### Options

+-------------------------+-------------+---------------------------------------------------------+
| Flag                    | Default     | Description                                             |
+=========================+=============+=========================================================+
| `--verbose` / `-v`      | off         | Show per-iteration details (hints, source counts, etc.) |
+-------------------------+-------------+---------------------------------------------------------+
| `--max-iter N` / `-n N` | from config | Override the max number of iterations                   |
+-------------------------+-------------+---------------------------------------------------------+

The output panel shows the inferred **location**, **confidence** (green ≥70%, yellow ≥40%, red <40%), **reasoning**, and **sources**.

## Data Sources (Gatherers)

+-----------+---------------------------+---------------------------------------------+
| Gatherer  | Service                   | Auth Required                               |
+===========+===========================+=============================================+
| News      | DuckDuckGo News           | No                                          |
+-----------+---------------------------+---------------------------------------------+
| Wikipedia | Wikipedia & Wikidata APIs | No                                          |
+-----------+---------------------------+---------------------------------------------+
| Events    | DuckDuckGo Web Search     | No                                          |
+-----------+---------------------------+---------------------------------------------+
| Twitter   | Twitter API v2            | `twitter_bearer_token`                      |
+-----------+---------------------------+---------------------------------------------+
| Reddit    | Reddit API                | `reddit_client_id` + `reddit_client_secret` |
+-----------+---------------------------+---------------------------------------------+
| Instagram | Instagram Graph API       | `instagram_access_token`                    |
+-----------+---------------------------+---------------------------------------------+

Gatherers that lack the required credentials are skipped automatically. The tool works with only the unauthenticated gatherers (News, Wikipedia, Events), but more sources improve accuracy.

## LLM

Tabber supports two providers, configured via `llm_provider`:

+--------------------+-------------------+---------------------+
| Provider           | Model             | Key                 |
+====================+===================+=====================+
| `openai` (default) | `gpt-4o`          | `openai_api_key`    |
+--------------------+-------------------+---------------------+
| `anthropic`        | `claude-opus-4-6` | `anthropic_api_key` |
+--------------------+-------------------+---------------------+

Both providers are accessed via the OpenAI-compatible SDK.

## Project Structure

```
tabber.py                  # CLI entry point
config.py                  # Config management
llm.py                     # LLM provider abstraction
models.py                  # Pydantic data models
gatherers/
    base.py                # Abstract base gatherer
    news.py
    wikipedia.py
    events.py
    twitter.py
    reddit.py
    instagram.py
modules/
    identification.py      # Disambiguation + feedback loop
    information_gathering.py  # Parallel gatherer orchestration
    location_analysis.py   # Final LLM location synthesis
```
