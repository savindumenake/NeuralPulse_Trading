"""
graph_builder.py - Conditional LangGraph StateGraph Construction (Phase 3)

Builds the deterministic Multi-Agent State-Machine using LangGraph.
Uses conditional routing after `tech_agent` so that fundamental news checks,
risk sizing, and Telegram signal broadcasts ONLY run when a high-conviction
technical trade setup (BUY or SELL) is detected:

    START -> tech_agent --(HOLD)--> END (Silent Watch)
                        --(BUY/SELL)--> sentiment_agent -> risk_arbiter -> execution_agent -> END
"""

from langgraph.graph import StateGraph, END

from agents.state import TradingState
from agents.technical_agent import technical_analyst_agent
from agents.sentiment_agent import sentiment_analyst_agent
from agents.risk_arbiter import risk_arbiter_agent
from agents.execution_agent import execution_agent


def _route_after_technical(state: TradingState) -> str:
    """
    Pro-Trader Gate: Only wake up Fundamental Sentiment & Risk Arbiter
    when Technical Confluence produces an actionable BUY or SELL signal.
    """
    if state.get("technical_signal") in ("BUY", "SELL"):
        return "sentiment_agent"
    return END


def build_trading_graph():
    """
    Construct the conditional trading pipeline as a LangGraph StateGraph.
    """
    graph = StateGraph(TradingState)

    # Register agent nodes
    graph.add_node("tech_agent", technical_analyst_agent)
    graph.add_node("sentiment_agent", sentiment_analyst_agent)
    graph.add_node("risk_arbiter", risk_arbiter_agent)
    graph.add_node("execution_agent", execution_agent)

    # Entry point -> Technical Analyst
    graph.set_entry_point("tech_agent")

    # Conditional gate: only proceed if technical setup is BUY or SELL
    graph.add_conditional_edges(
        "tech_agent",
        _route_after_technical,
        {
            "sentiment_agent": "sentiment_agent",
            END: END,
        },
    )
    graph.add_edge("sentiment_agent", "risk_arbiter")
    graph.add_edge("risk_arbiter", "execution_agent")
    graph.add_edge("execution_agent", END)

    return graph.compile()
