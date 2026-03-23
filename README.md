# Tabber

An OSINT CLI tool that determines the most likely current or recent physical location of a public figure by aggregating data from multiple sources and reasoning over them with an LLM.

## How It Works

1. **Disambiguation** — The input name is resolved to a structured person profile via LLM.
2. **Feedback loop** (up to N iterations):
   - The LLM generates targeted search hints based on the profile and any prior data.
   - All configured gatherers run in parallel to collect raw data.
   - The LLM evaluates whether there is sufficient location signal; exits early if so, or refines and repeats.
3. **Location analysis** — All gathered data is synthesised by a final LLM call into a location result with confidence and reasoning.
4. Results are displayed in a Rich terminal panel with colour-coded confidence and **automatically cached** to SQLite for instant recall on repeat lookups.

## Installation

```bash
git clone https://github.com/itsnotqwerty/tabber.git
cd tabber
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

To also enable the REST API server:

```bash
pip install -e ".[server]"
```

## Configuration

Configuration is stored in `~/.tabber/config.json` and managed via the `config` subcommand.

```bash
tabber config set <key> <value>
tabber config show
```

### Configuration Keys

| Key                      | Default                 | Description                                  |
| ------------------------ | ----------------------- | -------------------------------------------- |
| `max_iterations`         | `3`                     | Max feedback loop iterations                 |
| `llm_provider`           | `openai`                | LLM backend to use (`openai` or `anthropic`) |
| `openai_api_key`         | —                       | Required when `llm_provider` is `openai`     |
| `anthropic_api_key`      | —                       | Required when `llm_provider` is `anthropic`  |
| `twitter_bearer_token`   | —                       | Enables the Twitter gatherer                 |
| `instagram_access_token` | —                       | Enables the Instagram gatherer               |
| `reddit_client_id`       | —                       | Required (with secret) for Reddit gatherer   |
| `reddit_client_secret`   | —                       | Required (with ID) for Reddit gatherer       |
| `cache_ttl_hours`        | `24`                    | How long a cached result stays valid (hours) |
| `db_path`                | `~/.tabber/results.db`  | SQLite database file location                |
| `server_host`            | `127.0.0.1`             | Default bind host for `tabber server`        |
| `server_port`            | `8000`                  | Default bind port for `tabber server`        |

At minimum, set your API key for the chosen provider:

```bash
# OpenAI (default)
tabber config set openai_api_key sk-...

# Anthropic
tabber config set llm_provider anthropic
tabber config set anthropic_api_key sk-ant-...
```

## Usage

```bash
tabber lookup "Elon Musk"
tabber "Elon Musk"        # shorthand
```

### Options

| Flag                    | Default     | Description                                           |
| ----------------------- | ----------- | ----------------------------------------------------- |
| `--verbose` / `-v`      | off         | Show per-iteration details (hints, source counts)     |
| `--max-iter N` / `-n N` | from config | Override the max number of iterations                 |
| `--no-cache`            | off         | Skip the cache and always run a fresh lookup          |

The output panel shows the inferred **location**, **confidence** (green ≥70%, yellow ≥40%, red <40%), **reasoning**, and **sources**. Results marked `(cached)` were served from the local database without making any LLM or network calls.

## Caching

Every completed lookup is stored in a local SQLite database (`~/.tabber/results.db`). On subsequent lookups for the same name, the cached result is returned immediately if it was created within the last `cache_ttl_hours` hours (default: 24).

```
~/.tabber/
├── config.json     # configuration
└── results.db      # SQLite result cache
```

### Cache behaviour

- **CLI** — cache is checked automatically before running the pipeline. Use `--no-cache` to force a fresh run (the new result is still stored).
- **API** — same logic: set `"no_cache": true` in the request body to bypass the cache.
- **TTL** — configure how long results stay valid: `tabber config set cache_ttl_hours 48`
- **Invalidation** — delete cached results for a name via the API (`DELETE /results/{name}`) or by removing the database file.

## REST API Server

Start the server with:

```bash
tabber server
```

### Server options

| Flag       | Default     | Description                            |
| ---------- | ----------- | -------------------------------------- |
| `--host`   | `127.0.0.1` | Bind address                           |
| `--port`   | `8000`      | Bind port                              |
| `--reload` | off         | Auto-reload on code changes (dev mode) |

The server uses FastAPI and requires the `[server]` extra (`pip install -e ".[server]"`). Interactive API docs are available at `http://localhost:8000/docs` once the server is running.

### Endpoints

| Method   | Path              | Description                                                          |
| -------- | ----------------- | -------------------------------------------------------------------- |
| `GET`    | `/health`         | Health check — returns `{"status": "ok"}`                            |
| `POST`   | `/lookup`         | Run or recall a lookup. Request: `{"name": str, "no_cache": bool}`   |
| `GET`    | `/results`        | List all stored results, newest first. Supports `?limit=N` (max 500) |
| `GET`    | `/results/{name}` | Most recent stored result for a name                                 |
| `DELETE` | `/results/{name}` | Invalidate all cached results for a name                             |

### Request / response examples

**POST /lookup**

```json
// request
{ "name": "Taylor Swift" }

// response
{
  "query_name": "Taylor Swift",
  "canon_name": "Taylor Swift",
  "cached": false,
  "timestamp": "2026-03-22T14:00:00+00:00",
  "result": {
    "location": "Nashville, Tennessee, USA",
    "confidence": 0.82,
    "reasoning": "Multiple recent news sources confirm a studio session in Nashville.",
    "sources": ["news", "wikipedia"]
  }
}
```

**GET /results?limit=5**

```json
[
  {
    "id": 3,
    "query_name": "Taylor Swift",
    "canon_name": "Taylor Swift",
    "location": "Nashville, Tennessee, USA",
    "confidence": 0.82,
    "reasoning": "...",
    "sources": ["news", "wikipedia"],
    "timestamp": "2026-03-22T14:00:00+00:00"
  }
]
```

**DELETE /results/Taylor%20Swift**

```json
{ "deleted": 1, "name": "Taylor Swift" }
```

### Error responses

| Status | Condition                                  |
| ------ | ------------------------------------------ |
| `422`  | Validation error (e.g. empty `name` field) |
| `502`  | Upstream error from LLM or gatherers       |
| `404`  | No stored result found for the given name  |

## Data Sources (Gatherers)

| Gatherer  | Service                   | Auth Required                               |
| --------- | ------------------------- | ------------------------------------------- |
| News      | DuckDuckGo News           | No                                          |
| Wikipedia | Wikipedia & Wikidata APIs | No                                          |
| Events    | DuckDuckGo Web Search     | No                                          |
| Twitter   | Twitter API v2            | `twitter_bearer_token`                      |
| Reddit    | Reddit API                | `reddit_client_id` + `reddit_client_secret` |
| Instagram | Instagram Graph API       | `instagram_access_token`                    |

Gatherers that lack the required credentials are skipped automatically. The tool works with only the unauthenticated gatherers (News, Wikipedia, Events), but more sources improve accuracy.

## LLM

Tabber supports two providers, configured via `llm_provider`:

| Provider           | Model             | Key                 |
| ------------------ | ----------------- | ------------------- |
| `openai` (default) | `gpt-4o`          | `openai_api_key`    |
| `anthropic`        | `claude-opus-4-6` | `anthropic_api_key` |

Both providers are accessed via the OpenAI-compatible SDK.

All LLM calls use **Pydantic structured outputs** via the `response_format` parameter so responses are parsed and validated automatically:

- **OpenAI** — uses `client.beta.chat.completions.parse(response_format=Model)`, which returns a validated Pydantic instance directly.
- **Anthropic** — passes the model's JSON schema via `response_format={"type": "json_schema", ...}` and validates the response with `Model.model_validate_json()`.

### Data Models (`models.py`)

| Model              | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `PersonProfile`    | Disambiguated identity — name, aliases, roles              |
| `HintsList`        | Wrapper for the list of LLM-generated search hints         |
| `GathererResult`   | Raw output from one data source                            |
| `OSINTBundle`      | Aggregated results across all gatherers for one iteration  |
| `SignalEvaluation` | LLM confidence score + reasoning for location sufficiency  |
| `LocationResult`   | Final inferred location with confidence and evidence trail |
| `LookupResponse`   | API response envelope — wraps `LocationResult` with metadata (query name, canonical name, cache flag, timestamp) |

## Testing

The test suite lives in `tests/` and uses [pytest](https://pytest.org). All LLM and external HTTP calls are mocked so tests run offline without API keys.

```bash
# Install dev dependencies (pytest + httpx for API tests)
pip install -e ".[dev]"

# Run the full suite
pytest

# Run with verbose output
pytest -v

# Run a specific file
pytest tests/test_llm.py
```

### Test coverage by file

| Test file                        | What it covers                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| `test_config.py`                 | `config.load`, `set_key`, `masked` — I/O redirected to `tmp_path`                   |
| `test_models.py`                 | Pydantic validation for every model, including `LookupResponse`                      |
| `test_llm.py`                    | `complete()` routing, system messages, `response_format` for both providers, missing-key errors |
| `test_identification.py`         | Each private function (`_disambiguate`, `_generate_hints`, etc.) and the full `run()` loop |
| `test_location_analysis.py`      | `analyse()` prompt construction and `response_format` pass-through                  |
| `test_information_gathering.py`  | Gatherer enable/disable logic and `gather()` bundle assembly                        |
| `test_gatherers.py`              | `is_configured` for every gatherer class, base class interface                      |
| `test_sqlite.py`                 | `init_db`, `save_result`, `get_latest`, `list_all`, `delete_by_name` — in-process SQLite |
| `test_caching.py`                | TTL expiry, `get_cached`, `store`, `invalidate` — DB redirected to `tmp_path`       |
| `test_api.py`                    | All five REST endpoints via `TestClient` — cached and fresh paths, error cases       |
