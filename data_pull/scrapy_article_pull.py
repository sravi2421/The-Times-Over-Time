import scrapy
import pandas as pd

class QuotesSpider(scrapy.Spider):
    name = "times_test"
    start_urls = pd.read_csv('~/src/git/The-Times-Over-Time/data/articles_87_99.csv')['web_url']
    def parse(self, response):
        if ('/ref/' in response.request.url) or ('gst/fullpage' in response.request.url):
            year = int(response.xpath('//p/time/text()').get()[-4:])
        else:
            year = int(response.request.url.split('nytimes.com/')[1][:4])
        
        if year<2013:
            text = ''.join(response.xpath('//p[@class="story-body-text story-content"]/text()').getall())
            yield {'url': response.request.url,
                    'year':year,
                    'article_text': text,
                    'title': response.xpath('//h1[@id="headline"]/text()').get()}
        else:
            text = ''.join(response.xpath('//section/div/div/p/text()').getall())
            yield {'url': response.request.url,
                    'year':year,
                    'article_text': text,
                    'title': response.xpath('//h1[@itemprop="headline"]/span/text()').get()}

                    