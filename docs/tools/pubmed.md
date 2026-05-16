# PubMedTool

Search the biomedical literature index at PubMed via NCBI's free E-utilities API.

## Quick start

```python
import asyncio
from effgen.tools.builtin import PubMedTool

async def main():
    tool = PubMedTool()
    result = await tool.execute(
        operation="search",
        query="transformer attention",
        max_results=5,
    )
    for paper in result.output["results"]:
        print(paper["title"], "-", paper["url"])

asyncio.run(main())
```

## Operations

| operation  | required params | description |
|------------|-----------------|-------------|
| `search`   | `query`         | Full-text search; returns PMIDs + summaries |
| `fetch`    | `pmid`          | Full XML metadata (title, authors, abstract, DOI, journal, year) |
| `abstract` | `pmid`          | Abstract text only |

`pmid` may be a single id (`"26952870"`) or a comma-separated list.

## Output schema

`search`:

```jsonc
{
  "query": "transformer attention",
  "count": 5,
  "total_count": 12345,
  "results": [
    {
      "pmid": "...",
      "title": "...",
      "authors": ["..."],
      "journal": "...",
      "pubdate": "...",
      "doi": "...",
      "url": "https://pubmed.ncbi.nlm.nih.gov/<pmid>/"
    }
  ],
  "source": "pubmed"
}
```

`fetch` / `abstract` return a similar envelope with a `results` list where each
record carries `pmid`, `title`, `authors`, `journal`, `year`, `abstract`,
`doi`, and `url`.

## Rate limits

NCBI E-utilities allow **3 req/sec without an API key**, or **10 req/sec** when
the `NCBI_API_KEY` environment variable is set. PubMedTool enforces both via a
process-local token bucket; no extra configuration needed.

## Notes

- No authentication required. The tool is registered automatically in the
  `research` preset.
- All HTTP calls go through `urllib.request`; no third-party deps.
