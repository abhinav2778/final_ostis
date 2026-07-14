# ─────────────────────────────────────────────────────────────────
# OSTIS Scrapy Settings
# ─────────────────────────────────────────────────────────────────

BOT_NAME = "ostis_crawler"

SPIDER_MODULES = ["ostis_crawler.spiders"]
NEWSPIDER_MODULE = "ostis_crawler.spiders"

ADDONS = {}

# ── Identity ───────────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

# ── Crawl Politeness ───────────────────────────────────────────────
ROBOTSTXT_OBEY           = True
DOWNLOAD_DELAY           = 3
RANDOMIZE_DOWNLOAD_DELAY = True
CONCURRENT_REQUESTS      = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 1

# ── Timeouts & Retries ─────────────────────────────────────────────
DOWNLOAD_TIMEOUT = 15
DNS_TIMEOUT      = 10
RETRY_TIMES      = 1

# ── AutoThrottle ───────────────────────────────────────────────────
AUTOTHROTTLE_ENABLED            = True
AUTOTHROTTLE_START_DELAY        = 3
AUTOTHROTTLE_MAX_DELAY          = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# ── Output ─────────────────────────────────────────────────────────
FEED_EXPORT_ENCODING = "utf-8"
