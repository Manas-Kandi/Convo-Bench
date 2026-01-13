# ConvoBench Frontend

A minimal, modern web interface for running and visualizing ConvoBench benchmarks.

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- NVIDIA API key (optional, for real models)

### Installation

```bash
# Install Python dependencies
cd ConversationalBench
pip install -e .

# Install frontend dependencies
cd frontend
npm install
```

### Running

**Option 1: Start script (recommended)**

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

**Option 2: Manual start**

Terminal 1 - API Server:
```bash
cd ConversationalBench
uvicorn api.main:app --reload --port 8000
```

Terminal 2 - Frontend:
```bash
cd ConversationalBench/frontend
npm run dev
```

Open http://localhost:3000

## Features

### Scenario Selection

- 12 benchmark scenarios across 4 categories
- Visual category indicators (relay, planning, coordination, adversarial)
- One-click selection

### Configuration

- **Chain Length**: 2-5 agents
- **Complexity**: simple, medium, complex
- **Runs**: 1, 5, 10, or 20 iterations
- **Evaluation**: Toggle LLM-based evaluation

### Model Selection

- Mock agents (no API key required)
- NVIDIA NIM models (requires NVIDIA_API_KEY)
- Visual indicators for reasoning-capable models

### Real-time Conversation View

- Live streaming of agent interactions
- Input/output for each step
- Duration metrics
- Run selector to view different iterations

### Evaluation Panel

- Overall score (0-5)
- Per-dimension breakdowns
- Score bars with color coding
- Run history visualization

## Architecture

```
frontend/
├── src/
│   ├── app/
│   │   ├── globals.css      # Tailwind + custom styles
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main dashboard
│   ├── components/
│   │   ├── ScenarioCard.tsx
│   │   ├── ModelSelector.tsx
│   │   ├── ConfigPanel.tsx
│   │   ├── ConversationView.tsx
│   │   ├── EvaluationPanel.tsx
│   │   └── RunProgress.tsx
│   └── lib/
│       ├── api.ts           # API client
│       └── utils.ts         # Utilities
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scenarios` | GET | List available scenarios |
| `/models` | GET | List available models |
| `/runs` | POST | Start a new benchmark run |
| `/runs/{id}` | GET | Get run status and results |
| `/ws/{id}` | WS | Real-time run updates |

## Design Principles

1. **Minimal**: No unnecessary UI elements
2. **Dark theme**: Easy on the eyes
3. **Information density**: Show what matters
4. **Real-time**: Live updates via WebSocket
5. **Responsive**: Works on different screen sizes

## Customization

### Colors

Edit `frontend/src/app/globals.css`:

```css
:root {
  --background: #0a0a0a;
  --foreground: #fafafa;
  --accent: #3b82f6;
  /* ... */
}
```

### API URL

Edit `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
