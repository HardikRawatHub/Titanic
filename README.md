# 🚢 Titanic Chat Agent

> A conversational AI agent that answers natural-language questions about the
> Titanic dataset, generates live visualisations, and presents everything in a
> polished Streamlit chat interface.

---

## Architecture

```
┌─────────────────────┐        HTTP / JSON        ┌──────────────────────────┐
│  Streamlit Frontend │ ────────────────────────►  │  FastAPI Backend         │
│  frontend/app.py    │ ◄────────────────────────  │  backend/main.py         │
└─────────────────────┘     { text, image_b64 }    └────────────┬─────────────┘
                                                                 │
                                                   ┌────────────▼─────────────┐
                                                   │  LangChain ReAct Agent   │
                                                   │  backend/agent.py        │
                                                   └────────────┬─────────────┘
                                                                │ uses
                                              ┌─────────────────┼────────────────┐
                                              ▼                 ▼                ▼
                                       query_dataset   generate_chart    get_dataset_info
                                              │                 │
                                       data_loader.py    visualizer.py
                                       (pandas / seaborn) (matplotlib)
```

---

## Project Structure

```
titanic_agent/
├── .env.example          ← copy to .env and add your API key
├── requirements.txt
├── README.md
│
├── backend/
│   ├── main.py           ← FastAPI app (entry point)
│   ├── agent.py          ← LangChain agent + tool definitions
│   ├── data_loader.py    ← loads & enriches the Titanic dataset
│   └── visualizer.py     ← all chart generation (8 chart types)
│
└── frontend/
    └── app.py            ← Streamlit chat UI
```

---

## Quick Start

### 1. Clone / unzip the project

```bash
cd titanic_agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in your **OpenAI API key**:

```
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-4o-mini
BACKEND_URL=http://localhost:8000
```

> **Where to get an OpenAI API key:**
> Go to https://platform.openai.com/api-keys → "Create new secret key"
> `gpt-4o-mini` is the cheapest model and works great for this project.

### 5. Start the FastAPI backend

Open a **first terminal** and run:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Verify it's working by opening http://localhost:8000/health in your browser.

### 6. Start the Streamlit frontend

Open a **second terminal** and run:

```bash
cd frontend
streamlit run app.py
```

Streamlit will open http://localhost:8501 automatically.

---

## Features

| Feature | Description |
|---|---|
| 💬 Natural language Q&A | Ask anything in plain English |
| 📊 8 chart types | Histograms, heatmaps, violin plots, pie charts, dashboards |
| ⬇️ Chart download | Download any chart as a PNG |
| 📋 Quick questions | One-click suggested questions in sidebar |
| 🖼️ Direct chart access | Click chart buttons in sidebar, bypass the agent |
| ⏱️ Latency display | Shows response time for each query |
| 🌑 Dark theme | Professional dark UI throughout |
| 🔄 Chat history | Agent remembers context across turns |

---

## Example Questions to Try

```
What percentage of passengers were male?
Show me a histogram of passenger ages
What was the average ticket fare?
How many passengers embarked from each port?
What was the survival rate by class?
How did family size affect survival?
Show me the class and gender survival heatmap
Tell me about the youngest passengers
Which port had the highest survival rate?
```

---

## Available Charts

| Chart Key | Description |
|---|---|
| `overview_dashboard` | 6-panel summary dashboard |
| `age_histogram` | Age distribution with mean/median lines |
| `survival_by_sex` | Counts and rates split by gender |
| `fare_distribution` | Fare histogram (full + clipped) |
| `embarkation_counts` | Bar + pie chart by port |
| `class_survival_heatmap` | Heatmap: survival % by class × sex |
| `family_size_survival` | Survival rate vs. family size bar chart |
| `age_survival_violin` | Violin plot: age by survival outcome |

---

## API Reference

The FastAPI backend exposes these endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/dataset/info` | Dataset metadata + stats |
| POST | `/dataset/chart` | Get a specific chart directly |
| POST | `/chat` | Main agent chat endpoint |

Interactive API docs: http://localhost:8000/docs

---

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn
- **Agent:** LangChain (`create_openai_tools_agent`), OpenAI GPT-4o-mini
- **Data:** pandas, seaborn (Titanic dataset built-in)
- **Charts:** matplotlib, seaborn
- **Frontend:** Streamlit
- **Config:** python-dotenv

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Backend offline" in Streamlit | Start `uvicorn main:app --reload` in the backend/ directory |
| `OPENAI_API_KEY not set` error | Make sure `.env` exists with a valid key |
| Charts not showing | Check the backend terminal for matplotlib errors |
| Slow responses | Try `gpt-3.5-turbo` in `.env` for faster (less accurate) responses |
| Port 8000 in use | Use `uvicorn main:app --port 8001` and update `BACKEND_URL` in `.env` |

---

## Extra Features Added (Beyond Requirements)

1. **8 chart types** including a 6-panel overview dashboard
2. **Direct chart sidebar** — click to render without typing
3. **Suggested questions panel** — one-click exploration
4. **Chat history context** — agent remembers previous turns
5. **Chart download button** — save any chart as PNG
6. **Derived dataset columns** — age groups, family size, is_alone, fare per person
7. **Response latency display** — transparency about speed
8. **Full FastAPI Swagger docs** — at `/docs` for easy testing
9. **Dark-themed professional UI** with custom CSS
10. **CORS enabled** — backend can be deployed separately

---

*Built with ❤️ using LangChain, FastAPI, and Streamlit.*
