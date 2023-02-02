import logging
import re
from typing import Optional, List

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger('furniture-trf-logger')


class WebCrawler:
    """A WebCrawler based on requests and BS4 which intercepts interesting HTML tags
    """
    A_TAG = 'a'

    def __init__(self, interesting_tags: List[str], timeout: int):
        """Creates a WebCrawler instance

        Args:
            interesting_tags (List[str]): list of interesting HTML arguments which should be logged by the crawler
            timeout (int): timeout value for http(s) connections
        """
        self.interesting_tags = interesting_tags
        if not self.A_TAG in self.interesting_tags:
            self.interesting_tags = interesting_tags + [self.A_TAG]

        self.timeout = timeout

        self.headers = requests.utils.default_headers()
        self.headers.update(
            {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36'})

    def get_hostname(self, addr: str) -> Optional[str]:
        """Retrieves the hostname and tld from a given URL (relative or absolute)

        Args:
            addr (str): target address (relative or absolute)

        Returns:
            Optional[str]: the hostname, if found - otherwise None
        """
        hostnames = re.findall('((?:[a-zA-Z0-9\-]+\.)+\w+)', addr)

        if len(hostnames) == 0:
            return None

        return hostnames[0]

    def convert_rel_to_abs(self, rel_addr: str, parent_scheme: str, parent_hostname: str) -> str:
        """Converts a relative URL to its absolute counterpart

        Args:
            rel_addr (str): the relative address
            parent_scheme (str): parent scheme (https or http or other)
            parent_hostname (str): parent hostname

        Returns:
            str: the absolute URL
        """
        # ensure '/about' becomes 'about'
        while len(rel_addr) > 0 and rel_addr[0] == '/':
            rel_addr = rel_addr[1:]

        # map 'about' to 'https://sth.tld/about'
        abs_addr = f'{parent_scheme}://{parent_hostname}/{rel_addr}'
        logger.info(f"Converted [{rel_addr}] -> [{abs_addr}]")

        return abs_addr

    def crawl(self, url: str, max_depth: int = 0, max_urls: Optional[int] = None) -> List[dict]:
        """Crawls the targeted URL and continues by discovering links in the given page

        Args:
            url (str): target URL for crawling
            max_depth (int, optional): the maximum depth for BFS crawling. Defaults to 0.
            max_urls (Optional[int], optional): a maximum number of URLS to discover and crawl automatically. Defaults to None.

        Returns:
            List[dict]: a list of dictionaries which contain per-URL crawl information (tags, values, etc.) 
        """
        num_urls = 0

        urls_queue = set()
        urls_queue.add((url, max_depth))
        visited = {}
        visited[url] = True

        parent_hostname = self.get_hostname(url)
        parent_scheme = 'https' if 'https' in url else 'http'

        results = []

        while len(urls_queue) > 0:
            (crt_url, crt_depth) = urls_queue.pop()

            crt_result = self.crawl_page(url=crt_url)

            num_urls += 1
            if max_urls is not None and num_urls >= max_urls:
                break

            if crt_depth > 0 and self.A_TAG in crt_result['tags']:

                for a_link in crt_result['tags'][self.A_TAG]:

                    crt_a_link = a_link
                    crt_a_link_hostname = self.get_hostname(crt_a_link)

                    # convert to absolute url format
                    if crt_a_link_hostname is None:
                        crt_a_link = self.convert_rel_to_abs(
                            crt_a_link, parent_scheme, parent_hostname)

                    if not crt_a_link in visited:
                        if self.get_hostname(crt_a_link) == parent_hostname:
                            urls_queue.add((crt_a_link, crt_depth - 1))
                            visited[crt_a_link] = True

            results.append(crt_result)

        return results

    def crawl_page(self, url: str) -> dict:
        """Crawls an individual page for tags of interest and generates a crawl report

        Args:
            url (str): the target URL address

        Returns:
            dict: a crawl report which includes tags, values, etc.
        """

        result = {'url': url, 'status': None, 'tags': {}}

        logger.info(f'Crawling URL: [{url}]')

        # queries the target URL
        response = None
        try:
            response = requests.get(url, verify=False, headers=self.headers, timeout = self.timeout)
        except Exception as request_exception:
            logger.warning(
                f'Failed to crawl [{url}]: {str(request_exception)}')
            return result

        soup = BeautifulSoup(response.text, "html.parser")

        # locates HTML tags of interest within the page and extracts values
        possible_tags = soup.find_all(self.interesting_tags)

        for possible_tag in possible_tags:
            tag_name = possible_tag.name
            tag_text = possible_tag.text.strip()

            if possible_tag.name == self.A_TAG:
                href_location = possible_tag.get('href')
                if href_location is None:
                    continue
                tag_text = href_location

            if not tag_name in result['tags']:
                result['tags'][tag_name] = []

            result['status'] = response.status_code
            result['tags'][tag_name].append(tag_text)

        return result
