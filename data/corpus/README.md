# Offline demo corpus

⚠️ **These documents are synthetic.** They were written for this repository so
that Muhaqqiq can be demonstrated, tested and graded with no API keys and no
network access. Names, figures and dates inside them are illustrative and must
not be quoted as real-world facts.

Every record carries `"synthetic": true`, and any report produced from this
corpus is stamped with a provenance banner saying so. This is deliberate: an
agent whose whole purpose is citation discipline should not be the thing that
launders made-up numbers into a document.

To research the real web instead:

```bash
MUHAQQIQ_SEARCH_PROVIDER=tavily TAVILY_API_KEY=... muhaqqiq research "..."
```

## Format

Each file is a JSON array of documents:

```json
{
  "doc_id": "unique-slug",
  "title": "Document title",
  "url": "https://example.org/...",
  "publisher": "Publisher name",
  "published": "2025-11-02",
  "credibility": "high | medium | low",
  "tags": ["keyword", "keyword"],
  "synthetic": true,
  "content": "Body text. Split into sentences by the retriever."
}
```

Drop another `.json` file in this folder and it is picked up automatically — no
code change needed.
