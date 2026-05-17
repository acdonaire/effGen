"""Live validation for RSSFeedTool + NewsTool."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from effgen.tools.builtin import NewsTool, RSSFeedTool

ROOT = Path(__file__).parent.parent

OUTPUTS = ROOT / "build_plan" / "v0.2.5" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

BBC_RSS = "http://feeds.bbci.co.uk/news/world/rss.xml"
HTML_NOT_RSS = "https://example.com"


def load_env_without_printing_keys() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(Path.home() / ".effgen" / ".env", override=False)


async def validate_rss() -> dict:
    tool = RSSFeedTool()
    print("\n--- RSSFeedTool: fetch BBC World News ---")
    result = await tool.execute(operation="fetch", url=BBC_RSS)
    if not result.success:
        print(f"  FAIL: {result.error}")
        return {"success": False, "error": result.error}

    out = result.output
    print(f"  Feed: {out['feed']['title']}")
    print(f"  Entries fetched: {out['entry_count']}")
    if out["entry_count"] < 10:
        return {"success": False, "error": "BBC RSS returned fewer than 10 entries", "output": out}
    for e in out["entries"][:3]:
        print(f"    - {e['title'][:80]}")

    print("\n--- RSSFeedTool: malformed HTML feed ---")
    bad = await tool.execute(operation="fetch", url=HTML_NOT_RSS)
    if not bad.success:
        print(f"  FAIL: {bad.error}")
        return {"success": False, "error": bad.error}
    if bad.output["entries"]:
        return {"success": False, "error": "HTML URL unexpectedly returned feed entries"}
    print("  Returned empty entries without crashing")

    print("\n--- RSSFeedTool: search 'world' in BBC ---")
    sr = await tool.execute(operation="search_in_feed", url=BBC_RSS, query="world")
    if not sr.success:
        print(f"  search FAIL: {sr.error}")
    else:
        print(f"  Matches for 'world': {sr.output['entry_count']}")

    return {"bbc": out, "malformed": bad.output, "search": sr.output if sr.success else None}


async def validate_news() -> dict:
    tool = NewsTool()
    print("\n--- NewsTool: top_headlines without NEWS_API_KEY (RSS fallback) ---")
    saved_news_key = os.environ.pop("NEWS_API_KEY", None)
    result = await tool.execute(operation="top_headlines", max_results=10)
    if saved_news_key:
        os.environ["NEWS_API_KEY"] = saved_news_key
    if not result.success:
        print(f"  FAIL: {result.error}")
        return {"success": False, "error": result.error}

    out = result.output
    print(f"  Backend: {out['backend']}")
    print(f"  Articles: {out['count']}")
    if out["backend"] != "rss" or out["count"] < 10:
        return {"success": False, "error": "RSS fallback did not return >=10 articles", "output": out}
    for a in out["articles"][:3]:
        print(f"    [{a['source']}] {a['title'][:70]}")

    newsapi_out = None
    if saved_news_key:
        print("\n--- NewsTool: top_headlines with NEWS_API_KEY (NewsAPI) ---")
        keyed = await tool.execute(operation="top_headlines", max_results=5)
        if not keyed.success:
            return {"success": False, "error": keyed.error}
        newsapi_out = keyed.output
        print(f"  Backend: {newsapi_out['backend']}")
        if newsapi_out["backend"] != "newsapi" or newsapi_out["count"] < 1:
            return {"success": False, "error": "NEWS_API_KEY is set but NewsAPI did not return results", "output": newsapi_out}

    print("\n--- NewsTool: top_headlines (technology) ---")
    tech = await tool.execute(operation="top_headlines", category="technology", max_results=5)
    if tech.success:
        print(f"  Tech articles: {tech.output['count']}")

    print("\n--- NewsTool: search 'AI' ---")
    sr = await tool.execute(operation="search", query="AI", max_results=10)
    if not sr.success:
        print(f"  search FAIL: {sr.error}")
    else:
        print(f"  Search results for 'AI': {sr.output['count']}")

    return {
        "rss_fallback": out,
        "newsapi": newsapi_out,
        "technology": tech.output if tech.success else None,
        "search": sr.output if sr.success else None,
    }


def validate_research_preset_agent() -> dict:
    print("\n--- Research preset agent: tech news today ---")
    if not os.environ.get("OPENAI_API_KEY"):
        return {"success": False, "error": "OPENAI_API_KEY not set"}

    from effgen.core.agent import AgentMode
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.presets import create_agent

    model = OpenAIAdapter(model_name="gpt-4o-mini", timeout=60, max_retries=2)
    model.load()
    agent = create_agent("research", model, max_iterations=4, enable_memory=False)
    try:
        result = agent.run(
            "Use the news tool to get today's technology news and summarize three headlines.",
            mode=AgentMode.SINGLE,
        )
        return {
            "success": result.success and result.tool_calls >= 1 and bool(result.output.strip()),
            "tool_calls": result.tool_calls,
            "output": result.output,
            "metadata": result.metadata,
        }
    finally:
        agent.close()


async def main():
    load_env_without_printing_keys()
    print("=" * 60)
    print("RSS + News Tools Validation")
    print("=" * 60)

    rss_out = await validate_rss()
    news_out = await validate_news()
    agent_out = validate_research_preset_agent()

    rss_path = OUTPUTS / "2-rss.json"
    news_path = OUTPUTS / "2-news.json"
    agent_path = OUTPUTS / "2-agent.json"
    verify_path = OUTPUTS / "verify-p2.md"

    with open(rss_path, "w") as f:
        json.dump(rss_out, f, indent=2, default=str)
    with open(news_path, "w") as f:
        json.dump(news_out, f, indent=2, default=str)
    with open(agent_path, "w") as f:
        json.dump(agent_out, f, indent=2, default=str)

    print("\nOutputs saved:")
    print(f"  {rss_path}")
    print(f"  {news_path}")
    print(f"  {agent_path}")

    rss_ok = rss_out.get("success", False) or isinstance(rss_out.get("bbc", {}).get("entries"), list)
    news_ok = news_out.get("success", False) or isinstance(news_out.get("rss_fallback", {}).get("articles"), list)
    agent_ok = bool(agent_out.get("success"))

    print(f"\nRSS OK: {rss_ok}")
    print(f"News OK: {news_ok}")
    print(f"Agent OK: {agent_ok}")

    with open(verify_path, "w") as f:
        f.write("# RSS + News Verification\n\n")
        f.write(f"- RSSFeedTool BBC entries >= 10: {rss_ok}\n")
        f.write(f"- NewsTool RSS fallback articles >= 10: {news_ok}\n")
        f.write(f"- Research preset agent called NewsTool: {agent_ok}\n")
        f.write(f"- NewsAPI backend checked: {bool(news_out.get('newsapi'))}\n")

    if not (rss_ok and news_ok and agent_ok):
        sys.exit(1)

    print("\nAll validations PASSED.")


if __name__ == "__main__":
    asyncio.run(main())
