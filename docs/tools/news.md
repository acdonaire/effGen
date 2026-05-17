# NewsTool

Aggregate news headlines from reputable sources using curated RSS feeds, with optional NewsAPI.org integration.

**Module:** `effgen.tools.builtin.news`  
**Tool name:** `news`  
**Requires:** `feedparser` (for RSS backend)  
**Auth:** None required (RSS). Set `NEWS_API_KEY` for NewsAPI.org.

## Backends

| Condition | Backend |
|-----------|---------|
| `NEWS_API_KEY` env var set | NewsAPI.org (higher relevance, better search) |
| No key | Curated RSS feeds (free, no auth) |

Both backends fall back to each other automatically. If NewsAPI fails, RSS is used.

## Built-in Sources

The RSS backend includes feeds from:

| Category | Sources |
|----------|---------|
| `general` | BBC World News, BBC News, NPR, The Guardian, Al Jazeera |
| `technology` | Hacker News, TechCrunch, Wired, Ars Technica, MIT Technology Review |
| `science` | Nature, Science Daily, New Scientist |
| `business` | Financial Times, Bloomberg Markets, BBC Business |
| `health` | BBC Health, NPR Health |
| `sports` | BBC Sport, ESPN |

## Operations

| Operation | Description |
|-----------|-------------|
| `top_headlines` | Latest headlines (optionally filtered by category/region) |
| `search` | Search news by query across all or category-specific sources |

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `operation` | string | No | `top_headlines` | One of `top_headlines`, `search` |
| `query` | string | For `search` | — | Search query |
| `category` | string | No | — | `general`, `technology`, `science`, `business`, `health`, `sports` |
| `region` | string | No | — | `us`, `global` (RSS backend only) |
| `max_results` | integer | No | `20` | Max articles (1–100) |
| `sources` | string | No | — | NewsAPI source IDs, comma-separated (NewsAPI only) |

## Output Schema

```json
{
  "success": true,
  "operation": "top_headlines",
  "backend": "rss",
  "category": "technology",
  "region": null,
  "count": 10,
  "articles": [
    {
      "title": "AI breakthrough announced",
      "link": "https://...",
      "published": "Sat, 16 May 2026 10:00:00 +0000",
      "summary": "...",
      "source": "TechCrunch",
      "source_url": "https://..."
    }
  ],
  "data": {
    "articles": [
      {
        "title": "AI breakthrough announced",
        "link": "https://..."
      }
    ]
  },
  "error": null
}
```

## Examples

```python
import asyncio
from effgen.tools.builtin import NewsTool

tool = NewsTool()

# Top technology headlines
result = asyncio.run(tool.execute(
    operation="top_headlines",
    category="technology",
    max_results=10
))

# Search for AI news
result = asyncio.run(tool.execute(
    operation="search",
    query="artificial intelligence",
    max_results=15
))
```

## Setting up NewsAPI

1. Sign up at [newsapi.org](https://newsapi.org) (free tier: 100 requests/day)
2. Add to your `.env`:
   ```
   NEWS_API_KEY=your_key_here
   ```
3. effGen automatically detects and uses it

## Preset Integration

`news` is included in the **research** and **general** presets.
