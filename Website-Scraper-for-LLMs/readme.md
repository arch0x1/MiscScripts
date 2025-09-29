This script uses crawl4ai to crawl websites from a sitemap, intelligently extract meaningful content, and build a structured knowledge base ready for LLMs, RAG pipelines, or search systems.


output/<br>
  ├─ individual/     # Per-page JSON + Markdown<br>
  ├─ consolidated/   # Full dataset, LLM-ready file, quality subset, stats

  Primary use case was to feed an entire heavy website to a LLM so I could ask it specific questions. 
