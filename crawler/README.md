# Module 1 — Crawler

Scrapy-based crawler for cybersecurity threat intelligence sources, including
a dedicated parser for India's CERT-In advisories.

## Run

```bash
cd crawler/ostis_crawler
scrapy crawl security_blogs
```

Output: `storage/articles.jsonl` (one JSON object per line).

## Design notes

- **Incremental**: on startup, the spider loads every URL already present in
  `storage/articles.jsonl` into `self.seen_urls` and skips them — safe to
  re-run daily without duplicating articles.
- **CERT-In routing**: any URL under `cert-in.org.in` is routed to
  `parse_certin_index` / `parse_certin_advisory` instead of the generic
  article parser, so advisory ID, CVE list, and severity are extracted with
  CERT-In-specific patterns.
- **Domain blocklist**: `BLOCKED_DOMAINS` filters out login pages, job
  boards, and social media links picked up incidentally while crawling.
- **Politeness**: robots.txt obeyed, 2–3s delay, 1 concurrent request per
  domain, AutoThrottle enabled.
