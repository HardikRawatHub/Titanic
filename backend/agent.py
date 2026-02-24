"""
agent.py
--------
LangChain ReAct agent equipped with custom Titanic-analysis tools.
The agent receives a natural-language question, selects the right
tool(s), executes them, and returns a structured answer + optional
base64 chart.
"""

import os
import json
import textwrap
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from data_loader import get_df, get_summary_stats
from visualizer import CHART_REGISTRY

load_dotenv()

# â”€â”€ LLM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _get_llm():
    # First priority: Groq (OpenAI-compatible endpoint).
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ChatOpenAI(
            model=model,
            api_key=groq_api_key,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            temperature=0,
            streaming=False,
        )

    # Primary path: xAI (Grok) via OpenAI-compatible API
    xai_api_key = os.getenv("XAI_API_KEY")
    if xai_api_key:
        if xai_api_key.startswith("gsk_"):
            raise ValueError(
                "XAI_API_KEY looks like a Groq key (gsk_...). "
                "Use an xAI key from console.x.ai."
            )
        model = os.getenv("XAI_MODEL", os.getenv("OPENAI_MODEL", "grok-4-1-fast-non-reasoning"))
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        return ChatOpenAI(
            model=model,
            api_key=xai_api_key,
            base_url=base_url,
            temperature=0,
            streaming=False,
        )

    # Backward-compatible fallback: OpenAI env vars
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0, streaming=False)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Tool definitions
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@tool
def query_dataset(question: str) -> str:
    """
    Answer factual / statistical questions about the Titanic dataset using
    pandas. Returns a JSON string with 'answer' (human-readable) and
    'data' (optional dict with raw numbers).

    Good for: counts, percentages, averages, medians, comparisons,
    breakdowns by sex / class / port / age group.
    """
    df  = get_df()
    out = {}

    q = question.lower()

    # â”€â”€ Gender / sex â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if any(k in q for k in ["male", "female", "gender", "sex"]):
        total  = len(df)
        males  = int((df["sex"] == "male").sum())
        females = int((df["sex"] == "female").sum())
        out = {
            "answer": (f"Out of {total} passengers, {males} were male "
                       f"({males/total*100:.1f}%) and {females} were female "
                       f"({females/total*100:.1f}%)."),
            "data": {"total": total, "males": males, "females": females,
                     "male_pct": round(males/total*100, 2),
                     "female_pct": round(females/total*100, 2)},
        }

    # â”€â”€ Survival â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif any(k in q for k in ["surviv", "died", "death", "perish"]):
        total    = len(df)
        survived = int(df["survived"].sum())
        died     = total - survived
        rate     = survived / total * 100

        # By sex
        sex_sv = df.groupby("sex")["survived"].agg(["sum", "mean"]).round(4)
        # By class
        cls_sv = df.groupby("class_name")["survived"].agg(["sum", "mean"]).round(4)

        out = {
            "answer": (
                f"{survived} out of {total} passengers survived ({rate:.1f}%). "
                f"{died} passengers did not survive ({died/total*100:.1f}%). "
                f"Female survival rate: {sex_sv.loc['female','mean']*100:.1f}%, "
                f"Male: {sex_sv.loc['male','mean']*100:.1f}%. "
                f"1st class: {cls_sv.loc['1st Class','mean']*100:.1f}%, "
                f"2nd class: {cls_sv.loc['2nd Class','mean']*100:.1f}%, "
                f"3rd class: {cls_sv.loc['3rd Class','mean']*100:.1f}%."
            ),
            "data": {
                "total": total, "survived": survived, "died": died,
                "survival_rate_pct": round(rate, 2),
                "by_sex":   sex_sv.to_dict(),
                "by_class": cls_sv.to_dict(),
            },
        }

    # â”€â”€ Age â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif "age" in q:
        age = df["age"].dropna()
        grp = df["age_group"].value_counts().sort_index()
        out = {
            "answer": (
                f"Average passenger age: {age.mean():.1f} years. "
                f"Median: {age.median():.0f} years. "
                f"Youngest: {age.min():.0f}, Oldest: {age.max():.0f}. "
                f"Age was missing for {df['age'].isna().sum()} passengers. "
                f"Breakdown â€” {', '.join(f'{k}: {v}' for k, v in grp.items())}."
            ),
            "data": {
                "mean": round(age.mean(), 2), "median": age.median(),
                "min": age.min(), "max": age.max(),
                "missing": int(df["age"].isna().sum()),
                "age_groups": grp.to_dict(),
            },
        }

    # â”€â”€ Fare / ticket price â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif any(k in q for k in ["fare", "ticket", "price", "cost", "paid"]):
        fare = df["fare"].dropna()
        cls_fare = df.groupby("class_name")["fare"].mean().round(2)
        out = {
            "answer": (
                f"Average ticket fare: Â£{fare.mean():.2f}. "
                f"Median: Â£{fare.median():.2f}. "
                f"Range: Â£{fare.min():.2f} â€“ Â£{fare.max():.2f}. "
                f"By class â€” 1st: Â£{cls_fare.get('1st Class', 0):.2f}, "
                f"2nd: Â£{cls_fare.get('2nd Class', 0):.2f}, "
                f"3rd: Â£{cls_fare.get('3rd Class', 0):.2f}."
            ),
            "data": {
                "mean": round(fare.mean(), 2), "median": round(fare.median(), 2),
                "min": round(fare.min(), 2), "max": round(fare.max(), 2),
                "by_class": cls_fare.to_dict(),
            },
        }

    # â”€â”€ Embarkation / port â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif any(k in q for k in ["embark", "port", "southampton", "cherbourg", "queenstown", "boarded"]):
        ports = df["embark_port"].value_counts().dropna()
        total = ports.sum()
        lines = ", ".join(f"{p}: {c} ({c/total*100:.1f}%)" for p, c in ports.items())
        out = {
            "answer": f"Passengers embarked from: {lines}. Total accounted for: {total}.",
            "data":   ports.to_dict(),
        }

    # â”€â”€ Class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif any(k in q for k in ["class", "pclass", "first", "second", "third"]):
        cls = df["class_name"].value_counts().sort_index()
        total = len(df)
        lines = ", ".join(f"{c}: {v} ({v/total*100:.1f}%)" for c, v in cls.items())
        out = {
            "answer": f"Passenger class breakdown â€” {lines}.",
            "data":   cls.to_dict(),
        }

    # â”€â”€ Family â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif any(k in q for k in ["family", "alone", "sibling", "spouse", "parent", "child"]):
        alone = int(df["is_alone"].sum())
        total = len(df)
        avg_fam = df["family_size"].mean()
        out = {
            "answer": (
                f"{alone} passengers ({alone/total*100:.1f}%) traveled alone. "
                f"Average family size: {avg_fam:.1f}. "
                f"Largest group: {df['family_size'].max()} members."
            ),
            "data": {
                "alone": alone, "with_family": total - alone,
                "avg_family_size": round(avg_fam, 2),
                "max_family_size": int(df["family_size"].max()),
            },
        }

    # â”€â”€ General dataset overview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    else:
        stats = get_summary_stats()
        out = {
            "answer": (
                f"The Titanic dataset contains {stats['total_passengers']} passengers. "
                f"Overall survival rate: {stats['survival_rate_pct']}%. "
                f"Average age: {stats['avg_age']} years. "
                f"Average fare: Â£{stats['avg_fare']}. "
                f"Male: {stats['male_count']}, Female: {stats['female_count']}."
            ),
            "data": stats,
        }

    return json.dumps(out)


@tool
def generate_chart(chart_type: str) -> str:
    """
    Generate a data visualization and return a base64-encoded PNG image.

    Available chart_type values (use EXACTLY as written):
    - age_histogram          â†’ histogram of passenger ages
    - survival_by_sex        â†’ survival counts and rates split by sex
    - fare_distribution      â†’ fare/ticket price distribution
    - embarkation_counts     â†’ passengers per embarkation port (bar + pie)
    - class_survival_heatmap â†’ heatmap of survival rate by class and sex
    - family_size_survival   â†’ survival rate vs. family size
    - age_survival_violin    â†’ age distribution violin plot by survival outcome
    - overview_dashboard     â†’ 6-panel summary dashboard

    Return value is a JSON string with keys 'chart_type' and 'image_b64'.
    """
    if chart_type not in CHART_REGISTRY:
        available = list(CHART_REGISTRY.keys())
        return json.dumps({"error": f"Unknown chart '{chart_type}'. Available: {available}"})

    df  = get_df()
    fn  = CHART_REGISTRY[chart_type]
    b64 = fn(df)
    return json.dumps({"chart_type": chart_type, "image_b64": b64})


@tool
def get_dataset_info() -> str:
    """
    Return a comprehensive overview of the Titanic dataset: shape,
    columns, missing values, and high-level summary statistics.
    Useful as a starting point or when the user asks 'what data do you have?'
    """
    df    = get_df()
    stats = get_summary_stats()
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    info = {
        "rows":              len(df),
        "columns":           list(df.columns),
        "missing_values":    {k: int(v) for k, v in missing.items()},
        "summary":           stats,
        "available_charts":  list(CHART_REGISTRY.keys()),
    }
    return json.dumps(info, default=str)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# System prompt
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

SYSTEM_PROMPT = textwrap.dedent("""
    You are TitanicBot ðŸš¢ â€” a friendly, knowledgeable data analyst specialising
    in the famous Titanic passenger dataset (891 records).

    ## Your personality
    - Warm, concise, slightly nautical in spirit.
    - Always cite numbers precisely (e.g. "74.2%" not "about 74%").
    - Add one interesting insight or "did you know?" fact when relevant.

    ## Tools at your disposal
    1. **query_dataset** â€“ for any factual / statistical question about passengers.
    2. **generate_chart** â€“ for any visualisation request. Pick the most relevant chart.
    3. **get_dataset_info** â€“ to learn about available columns, missing data, etc.

    ## Rules
    - Always call query_dataset or get_dataset_info BEFORE writing a statistical answer.
    - When the user asks for a visual / chart / plot / histogram / graph, call generate_chart.
    - If a question covers BOTH stats and a chart, call BOTH tools.
    - Respond in Markdown. Use bold for key numbers.
    - Keep answers under 200 words unless the user asks for detail.
    - Never make up numbers; always derive them from tool results.
    - NEVER mention tool names like get_dataset_overview, query_dataframe, or create_visualization to the user. Just answer naturally.
""").strip()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Build agent
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_agent() -> AgentExecutor:
    llm   = _get_llm()
    tools = [query_dataset, generate_chart, get_dataset_info]
    agent = create_openai_tools_agent(llm, tools, PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # prints tool calls in the terminal
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def run_agent(question: str, chat_history: list | None = None) -> dict:
    """
    Run the agent on a single question and return a structured dict:
    {
        "text":    str,           # Markdown answer
        "image_b64": str | None,  # base64 PNG if a chart was generated
        "chart_type": str | None
    }
    """
    executor = build_agent()
    try:
        result = executor.invoke({
            "input": question,
            "chat_history": chat_history or [],
        })
    except Exception:
        # Some OpenAI-compatible providers occasionally emit malformed tool-call
        # payloads (e.g., null tool arguments), which can raise parser errors
        # inside LangChain. Return a safe response instead of HTTP 500.
        return {
            "text": "I wasn't able to complete that analysis. Please rephrase your question or try a simpler query.",
            "image_b64": None,
            "chart_type": None,
        }

    image_b64 = None
    chart_type = None

    for step in (result.get("intermediate_steps") or []):
        action, observation = step
        if hasattr(action, "tool") and action.tool == "generate_chart":
            try:
                obs_data = json.loads(observation)
                if "image_b64" in obs_data:
                    image_b64 = obs_data["image_b64"]
                    chart_type = obs_data.get("chart_type")
            except Exception:
                pass

    return {
        "text": result.get("output") or "I wasn't able to complete that analysis. Could you rephrase your question?",
        "image_b64": image_b64,
        "chart_type": chart_type,
    }
