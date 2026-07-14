import scrapy


class OstisCrawlerItem(scrapy.Item):
    id          = scrapy.Field()
    title       = scrapy.Field()
    url         = scrapy.Field()
    content     = scrapy.Field()
    source      = scrapy.Field()
    advisory_id = scrapy.Field()
    cves        = scrapy.Field()
    severity    = scrapy.Field()
    affected    = scrapy.Field()
    crawled_at  = scrapy.Field()
