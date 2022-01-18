import requests
import re
import json
import time
import logging
from typing import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def print_json(obj):
    print("Width {0}, Height {1}".format(obj["width"], obj["height"]))
    print("Thumbnail {0}".format(obj["thumbnail"]))
    print("Url {0}".format(obj["url"]))
    print("Title {0}".format(obj["title"].encode('utf-8')))
    print("Image {0}".format(obj["image"]))
    print("__________")


def search(keywords) -> Iterator[dict]:
    url = 'https://duckduckgo.com/'
    params = {
        'q': keywords
    }

    logger.debug("Hitting DuckDuckGo for Token")

    #   First make a request to above URL, and parse out the 'vqd'
    #   This is a special token, which should be used in the subsequent request
    res = requests.post(url, data=params)
    search_obj = re.search(r'vqd=([\d-]+)&', res.text, re.M | re.I)

    if not search_obj:
        logger.error("Token Parsing Failed !")
        return -1

    logger.debug("Obtained Token")

    headers = {
        'authority': 'duckduckgo.com',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'sec-fetch-dest': 'empty',
        'x-requested-with': 'XMLHttpRequest',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/80.0.3987.163 Safari/537.36',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'referer': 'https://duckduckgo.com/',
        'accept-language': 'en-US,en;q=0.9',
    }

    params = (
        ('l', 'us-en'),
        ('o', 'json'),
        ('q', keywords),
        ('vqd', search_obj.group(1)),
        ('f', ',,,'),
        ('p', '1'),
        ('v7exp', 'a'),
    )

    request_url = url + "i.js"

    logger.debug("Hitting Url : %s", request_url)

    while True:
        while True:
            try:
                res = requests.get(request_url, headers=headers, params=params)
                data = json.loads(res.text)
                break
            except ValueError:
                logger.debug("Hitting Url Failure - Sleep and Retry: %s", request_url)
                time.sleep(5)
                continue

        logger.debug("Hitting Url Success : %s", request_url)
        yield from data["results"]

        if "next" not in data:
            logger.debug("No Next Page - Exiting")
            break

        request_url = url + data["next"]


def silhouette_urls_from_noun(noun: str) -> Iterator[str]:
    yield from (r["image"] for r in search(f"{noun} silhouette"))


if __name__ == '__main__':
    from itertools import islice

    for image_url in islice(silhouette_urls_from_noun('cat'), 5):
        print(image_url)
