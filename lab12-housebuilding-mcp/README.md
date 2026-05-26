# MAS House Building with MCP Multi-Agent Construction Contracting

A complete multi-agent simulation system modeling ACME outsourcing construction work to companies A–F using **real MCP (Model Context Protocol) servers**. Agents discover tools from servers and make LLM-driven decisions.

## How to Run

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Setup environment (one time)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
nano .env
```

**Start the MCP servers in separate terminals:**

```bash
# Terminal 1: Auction server (port 8010)
python -m auction_server.mcp_server

# Terminal 2: Negotiation server (port 8011)
python -m negotiation_server.mcp_server
```

**Then run the orchestrator:**

```bash
# Terminal 3: Run the simulation
python orchestrator.py
```

The orchestrator connects to the running FastMCP servers via HTTP/SSE at `localhost:8010` and `localhost:8011`, and runs the complete auction and negotiation simulation with detailed logging.

### MCP Servers

#### Auction Server (`auction_server/mcp_server.py`)
- **Transport**: HTTP/SSE on port 8010
- **Tools**:
  - `start_auction(task_name, budget)` — Initialize auction
  - `propose_budget(task_name, price)` — ACME proposes price
  - `bid(task_name, company)` — Company submits bid
  - `get_status(task_name)` — Get auction state
- **Protocol**: Fixed 3 rounds, companies bid YES/NO at ACME's proposed prices

#### Negotiation Server (`negotiation_server/mcp_server.py`)
- **Transport**: HTTP/SSE on port 8011
- **Tools**:
  - `start_negotiation(task_name, company_names, ...)` — Initialize negotiation
  - `make_offer(task_name, company, price, offer_type)` — Make/counter/accept offer
  - `get_status(task_name)` — Get negotiation state
- **Protocol**: Up to 3 rounds, monotonic concession (prices only decrease)

### Agents (LLM-Driven)

- **ACME Agent**: Reads tool descriptions from server, decides what price to propose and what offers to make
- **Company Agents**: 6 independent instances (A–F), each reads tools and decides whether to bid or what offer to make

**Key**: Agents do NOT hardcode tool names. Prompts show available tools, and LLM decides which tool to call.

## Project Structure

```
MAS-HouseBuilding-MCP/
├── auction_server/
│   ├── mcp_server.py      # Real MCP server with async handlers
│   ├── client.py          # Client adapter for in-process or stdio
│   ├── state.py           # AuctionState dataclass
│   └── tools.py           # AuctionTools (propose_budget, bid, get_status)
├── negotiation_server/
│   ├── mcp_server.py      # Real MCP server with async handlers
│   ├── client.py          # Client adapter
│   ├── state.py           # NegotiationState dataclass
│   └── tools.py           # NegotiationTools (make_offer, get_status)
├── agents/
│   ├── base_agent.py      # Base agent with OpenAI client
│   ├── acme.py            # ACME buyer agent workflows
│   └── company.py         # Company contractor agent workflows
├── prompts/
│   ├── acme_auction_round.txt          # ACME auction decisions
│   ├── company_auction_round.txt       # Company auction decisions
│   ├── acme_negotiation_round.txt      # ACME negotiation decisions
│   └── company_negotiation_round.txt   # Company negotiation decisions
├── config/
│   ├── llm_config.py      # LLM configuration (gpt-5-nano default)
│   ├── acme_config.py     # ACME tasks and budget
│   └── contractors.yaml   # Company specialties and costs
├── shared/
│   ├── types.py           # Dataclasses (Task, Bid, Offer, etc.)
│   └── logger.py          # Structured logging
├── orchestrator.py        # Main entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

## Run Simulation

After starting both servers in separate terminals, run the orchestrator:

```bash
python orchestrator.py
```

The orchestrator will:
1. Connect to the auction server at `http://127.0.0.1:8010/sse`
2. Connect to the negotiation server at `http://127.0.0.1:8011/sse`
3. Execute the complete auction and negotiation simulation
4. Log all decisions and outcomes with structured logging

## Configuration

Control LLM reasoning effort via environment variable:

```bash
# Medium reasoning (default)
python orchestrator.py
```

## How It Works

### Auction Phase (Per Task)

1. ACME agent calls `run_auction_workflow([task])`
2. Server initializes auction state
3. For 3 rounds:
   - ACME's LLM reads available tools → decides price to propose
   - Server executes `propose_budget(price)`
   - Each company's LLM reads available tools → decides to bid or skip
   - Companies that bid YES are added to bidders list
4. All companies that bid YES proceed to negotiation

### Negotiation Phase (Per Task-Bidders Pair)

1. ACME agent calls `run_negotiation_workflow(auction_bidders, [task])`
2. Server initializes negotiation state with all bidders
3. For up to 3 rounds:
   - ACME's LLM reads available tools → decides offer (price + type)
   - Server executes `make_offer(price, type)`
   - Each company's LLM reads available tools → decides offer
   - Server executes company's `make_offer(price, type)`
   - If any offer is `accept`, negotiation ends
4. Final price and contractor recorded

## Key Design Points

1. **Tool Discovery**: Agents receive tool descriptions in prompts, decide which tool to call
2. **Response Format**: All agent responses are JSON: `{"tool": "...", "arguments": {...}}`
3. **Server as Source of Truth**: All state lives on server, agents only read/execute
4. **Minimal Agent Memory**: Only `last_action` and `last_offer_seen`, rest from server
5. **Protocol Enforcement**: Server enforces auction/negotiation rules, agents can't bypass
6. **Full Logging**: Every action logged with before/after state for replay

## Testing

Run the simulation a few times with different `LLM_CONFIG` values. You should see:

- All 4 tasks run through auction and negotiation
- Multiple companies bidding on same task (both auction and negotiation)
- Final contracts within budget
- Total savings displayed

Example output:
```
=== AUCTION PHASE: structural design ===
--- Round 1 ---
ACME proposes: $3000.00
Company A: Bid submitted
Company C: Bid submitted
Auction complete. Bidders: ['A', 'C']

=== NEGOTIATION PHASE: structural design ===
Negotiating with: A, C
--- Negotiation Round 1 ---
ACME offer: $3000.00
A counter: $3200.00
C counter: $3100.00
...
```

## Environment

Create `.env` file:
```
OPENAI_API_KEY=sk-...
LLM_CONFIG=default
```

## Requirements

- Python 3.9+
- `openai` — For OpenAI API (gpt-5-nano, gpt-4-turbo)
- `mcp` — For MCP server/client
- `pyyaml` — For contractor config
- `python-dotenv` — For .env loading
