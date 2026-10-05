import html
import re
import urllib.parse
from html.parser import HTMLParser
from typing import List, Dict

class DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = None
        self.current_tag = None
        self.in_title = False
        self.in_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "").split()

        if tag == "div" and "result" in classes:
            if self.current_result and self.current_result.get("title") and self.current_result.get("url"):
                self.results.append(self.current_result)
            self.current_result = {"title": "", "url": "", "snippet": ""}

        if tag == "a" and "result__a" in classes:
            self.in_title = True
            if self.current_result is not None:
                self.current_result["url"] = attrs_dict.get("href", "")

        if tag == "a" and "result__snippet" in classes:
            self.in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a":
            self.in_title = False
            self.in_snippet = False

    def handle_data(self, data):
        if self.current_result is None:
            return
        if self.in_title:
            self.current_result["title"] += data
        elif self.in_snippet:
            self.current_result["snippet"] += data

    def finalize(self):
        if self.current_result and self.current_result.get("title") and self.current_result.get("url"):
            self.results.append(self.current_result)
        return self.results
