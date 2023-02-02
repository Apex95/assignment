import csv
import json
import sys
import logging
import urllib3

from inference import FurnitureTrfInterface
from crawler import WebCrawler

# stops warnings related to invalid SSL certificates
urllib3.disable_warnings()

logger = logging.getLogger('furniture-trf-logger')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


INTERESTING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5']
CLS_THRESHOLD = 0.5
KEEP_THRESHOLD = 0.6

furnitureTrf = FurnitureTrfInterface(
    device = 'cpu',
    backbone = 'distilbert-base-uncased',
    tokenizer = 'distilbert-base-uncased',
    checkpoint = 'checkpoints/4.168538576697756.dat'
)

crawler = WebCrawler(interesting_tags = INTERESTING_TAGS,
                     timeout = 10)

results_dict = {}

with open('data/furniture stores pages.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter='\n')
    
    # skip header
    next(csvreader)

    for row in csvreader:
        crawler_results = crawler.crawl(row[0], max_depth = 0, max_urls = 50)
        
        if not row[0] in results_dict:
            results_dict[row[0]] = []

        # for each crawled URL
        for crawler_result in crawler_results:
            found_product = False

            # check content of headings from most to least important (h1->h5)
            for k in INTERESTING_TAGS:
                
                # found possible product name in heading -> stopping
                if found_product:
                    break

                # if tag k is found within the site's source code
                if k in crawler_result['tags']:
                    # look at the values form all tag k's occurences
                    for e in crawler_result['tags'][k]:

                        # run inference step
                        trf_result = furnitureTrf.run_inference(e, threshold = CLS_THRESHOLD)
                        is_product = trf_result['output']
                        confidence = trf_result['confidence']

                        # save / log only products that match the confidence threshold
                        if is_product and confidence > KEEP_THRESHOLD:

                            if not crawler_result['url'] in results_dict:
                                results_dict[crawler_result['url']] = []

                            results_dict[crawler_result['url']].append({'query' : e, 
                                                                        'is_product' : is_product, 
                                                                        'confidence' : confidence})
                            found_product = True

        with open("outputs/out.json", 'w') as f:
            json.dump(results_dict, f)

        