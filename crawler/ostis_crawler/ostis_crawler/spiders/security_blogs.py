# ─────────────────────────────────────────────────────────────────
# OSTIS — Crawler Spider
# Crawls cybersecurity sources including India-specific CERT-In.
# Incremental — skips already-seen URLs recorded in storage/articles.jsonl.
# Domain blocklist — filters off-topic / non-article pages.
# ─────────────────────────────────────────────────────────────────

import scrapy
import hashlib
import json
import os
import re
from datetime import datetime


class SecurityBlogsSpider(scrapy.Spider):
    name = "security_blogs"

    custom_settings = {
        "DOWNLOAD_DELAY":                 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "ROBOTSTXT_OBEY":                 True,
        "AUTOTHROTTLE_ENABLED":           True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
    }

    start_urls = [
        # ── India — Government & CERT ─────────────────────────────
        "https://www.cert-in.org.in/s2cMainServlet?pageid=PUBVLNOTES01",
        "https://www.cert-in.org.in/s2cMainServlet?pageid=PUBADVSRY",
        "https://csirt.gov.in/advisories",
        "https://www.meity.gov.in/content/cyber-security",
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",

        # ── India — Cybersecurity News ────────────────────────────
        "https://economictimes.indiatimes.com/tech/technology/cybersecurity",
        "https://www.thehindu.com/sci-tech/technology/cybersecurity/",
        "https://www.livemint.com/technology/tech-news/cybersecurity",
        "https://www.business-standard.com/technology/tech-news/cyber-security",

        # ── General Cybersecurity News ─────────────────────────────
        "https://thehackernews.com/",
        "https://krebsonsecurity.com/",
        "https://www.bleepingcomputer.com/",
        "https://www.darkreading.com/",
        "https://www.infosecurity-magazine.com/news/",
        "https://cyberscoop.com/",
        "https://www.scmagazine.com/",
        "https://www.helpnetsecurity.com/",
        "https://threatpost.com/",
        "https://www.securityweek.com/",
        "https://www.securityaffairs.co/",
        "https://therecord.media/",

        # ── Threat Intelligence ────────────────────────────────────
        "https://unit42.paloaltonetworks.com/",
        "https://securelist.com/",
        "https://www.malwarebytes.com/blog/",
        "https://blog.talosintelligence.com/",
        "https://www.sentinelone.com/blog/",
        "https://www.crowdstrike.com/blog/",
        "https://www.mandiant.com/resources/blog",
        "https://research.checkpoint.com/",
        "https://www.proofpoint.com/us/blog",
        "https://www.rapid7.com/blog/",
        "https://www.tenable.com/blog",
        "https://www.recordedfuture.com/blog",

        # ── Government / CERT / National ──────────────────────────
        "https://www.cisa.gov/news-events/cybersecurity-advisories",
        "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
        "https://www.ncsc.gov.uk/news/all",
        "https://www.cyber.gov.au/about-us/news",
        "https://www.enisa.europa.eu/news",

        # ── Finance / Banking ──────────────────────────────────────
        "https://www.bankinfosecurity.com/",
        "https://www.fsisac.com/insights",
        "https://www.finextra.com/newsroom/cybersecurity",
        "https://www.pymnts.com/cybersecurity/",
        "https://www.americanbanker.com/cybersecurity",

        # ── Healthcare ──────────────────────────────────────────────
        "https://www.healthcareitnews.com/topic/cybersecurity",
        "https://healthitsecurity.com/news",
        "https://www.hipaajournal.com/",
        "https://www.healthcareinfosecurity.com/",
        "https://www.beckershospitalreview.com/cybersecurity.html",

        # ── Education ───────────────────────────────────────────────
        "https://er.educause.edu/topics/cybersecurity",
        "https://campustechnology.com/Articles/List/Security.aspx",
        "https://edscoop.com/cybersecurity/",
        "https://www.eschoolnews.com/category/security/",

        # ── ICS / OT / SCADA ──────────────────────────────────────────
        "https://www.dragos.com/blog/",
        "https://claroty.com/team82/research",
        "https://industrialcyber.co/",
        "https://www.nozominetworks.com/blog/",
        "https://www.forescout.com/blog/",

        # ── IoT Security ──────────────────────────────────────────────
        "https://www.iotsecurityfoundation.org/news/",
        "https://www.iottechnews.com/category/security/",
    ]

    BLOCKED_DOMAINS = {
        "careers.microsoft.com", "education.microsoft.com",
        "learn.microsoft.com", "docs.microsoft.com",
        "login.microsoftonline.com", "login.live.com",
        "marketplace.microsoft.com", "go.microsoft.com", "aka.ms",
        "twitter.com", "facebook.com", "linkedin.com",
        "youtube.com", "instagram.com", "reddit.com",
        "t.co", "bit.ly", "tinyurl.com",
        "amazon.com", "ebay.com", "walmart.com",
        "careers.google.com", "jobs.lever.co",
        "greenhouse.io", "workday.com",
        "play.google.com", "apps.apple.com",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
            )
        )
        self.output_path = os.path.join(BASE_DIR, "storage", "articles.jsonl")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.seen_urls = set()

        if os.path.exists(self.output_path):
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        self.seen_urls.add(record.get("url", ""))
                    except Exception:
                        pass
            self.logger.info(
                f"Loaded {len(self.seen_urls)} existing URLs — will skip these."
            )

    # ── Main router ────────────────────────────────────────────────
    def parse(self, response):
        """Route CERT-In pages to a dedicated parser; everything else generic."""
        if "cert-in.org.in" in response.url:
            yield from self.parse_certin_index(response)
        else:
            for link in response.css("a::attr(href)").getall():
                full_url = response.urljoin(link)
                if self._is_article(full_url):
                    yield scrapy.Request(
                        full_url,
                        callback=self.parse_article,
                        errback=self.handle_error,
                    )

    # ── CERT-In index parser ───────────────────────────────────────
    def parse_certin_index(self, response):
        """Follow advisory links from CERT-In listing/index pages."""
        for link in response.css(
            "a[href*='VLNOTE'], a[href*='ADVISORY'], "
            "a[href*='vlnotes'], a[href*='advisory']"
        ):
            href = link.attrib.get("href", "")
            full_url = response.urljoin(href)
            if full_url not in self.seen_urls:
                self.seen_urls.add(full_url)
                yield scrapy.Request(
                    full_url,
                    callback=self.parse_certin_advisory,
                    errback=self.handle_error,
                )

        for link in response.css(
            "a[href*='pageid=PUBVLN'], a[href*='pageid=PUBADV'], "
            "a[href*='pageid=PUBVLNOTES'], a[href*='pageid=PUBADVSRY']"
        ):
            href = link.attrib.get("href", "")
            full_url = response.urljoin(href)
            if full_url not in self.seen_urls:
                self.seen_urls.add(full_url)
                yield scrapy.Request(
                    full_url,
                    callback=self.parse_certin_index,
                    errback=self.handle_error,
                )

    # ── CERT-In advisory detail parser ────────────────────────────
    def parse_certin_advisory(self, response):
        """Parse a single CERT-In vulnerability note or advisory page."""
        title = (
            response.css("h2::text, h3::text, h4::text").get("") or
            response.css("title::text").get("") or
            ""
        ).strip()
        if not title or len(title) < 5:
            return

        raw_parts = response.css("td::text, p::text, li::text, span::text").getall()
        content = " ".join(p.strip() for p in raw_parts if len(p.strip()) > 15)
        if len(content) < 100:
            return

        advisory_id = ""
        aid_match = re.search(r'(CI(?:VN|AD)-\d{4}-\d{4,6})', content, re.IGNORECASE)
        if aid_match:
            advisory_id = aid_match.group(1).upper()

        cves = sorted(set(
            c.upper() for c in re.findall(r'CVE-\d{4}-\d{4,7}', content, re.IGNORECASE)
        ))

        severity = "UNKNOWN"
        sev_match = re.search(r'severity\s*(?:rating)?[:\s]+([A-Z]+)', content, re.IGNORECASE)
        if sev_match:
            word = sev_match.group(1).upper()
            if word in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
                severity = word

        affected = ""
        aff_match = re.search(
            r'software\s+affected[:\s]+(.{10,200}?)(?:\n|description|overview)',
            content, re.IGNORECASE | re.DOTALL
        )
        if aff_match:
            affected = aff_match.group(1).strip()[:200]

        article = {
            "id":          hashlib.md5(response.url.encode()).hexdigest(),
            "title":       title,
            "url":         response.url,
            "content":     content,
            "source":      "CERT-In",
            "advisory_id": advisory_id,
            "cves":        cves,
            "severity":    severity,
            "affected":    affected,
            "crawled_at":  datetime.utcnow().isoformat(),
        }

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(article) + "\n")

        yield article

    # ── Generic article parser ─────────────────────────────────────
    def parse_article(self, response):
        content_type = response.headers.get("Content-Type", b"").decode().lower()
        if "text/html" not in content_type:
            return

        title = (
            response.css('meta[property="og:title"]::attr(content)').get() or
            response.css("title::text").get() or
            response.css("h1::text").get() or
            ""
        ).strip()
        if not title or len(title) < 5:
            return

        paragraphs = response.css(
            "article p::text, .content p::text, .post-content p::text, "
            "main p::text, p::text"
        ).getall()
        content = " ".join(p.strip() for p in paragraphs if len(p.strip()) > 40)
        if len(content) < 200:
            return

        article = {
            "id":         hashlib.md5(response.url.encode()).hexdigest(),
            "title":      title,
            "url":        response.url,
            "content":    content,
            "crawled_at": datetime.utcnow().isoformat(),
        }

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(article) + "\n")

        yield article

    # ── URL filter ─────────────────────────────────────────────────
    def _is_article(self, url):
        url_lower = url.lower()
        try:
            domain = url.split("/")[2].lower()
            clean_domain = domain.replace("www.", "")
            if any(bd in clean_domain for bd in self.BLOCKED_DOMAINS):
                return False
        except Exception:
            return False

        skip_patterns = [
            "/tag/", "/tags/", "/category/", "/categories/",
            "/author/", "/page/", "/feed", "/rss", "/atom",
            "/login", "/signup", "/register", "/subscribe",
            "/cart", "/checkout", "/shop", "/store", "/product",
            "/about", "/contact", "/careers", "/jobs", "/team",
            "/events/", "/webinar", "/conference", "/workshop",
            "/privacy", "/terms", "/legal", "/cookie", "/gdpr",
            "/advertise", "/sponsor", "/partner",
            "/sitemap", "/robots",
            ".pdf", ".zip", ".png", ".jpg", ".jpeg",
            ".gif", ".svg", ".mp4", ".mp3",
            "javascript:", "mailto:", "tel:", "#",
        ]
        if any(p in url_lower for p in skip_patterns):
            return False
        if url in self.seen_urls:
            return False

        self.seen_urls.add(url)
        return True

    def handle_error(self, failure):
        self.logger.warning(f"Request failed: {failure.request.url}")
