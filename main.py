#! /usr/bin/env python
"""
This module implements the document processing portion of a full-text ad-hoc search engine for racket documentation.
It specifically targets the Racket Reference.
"""

import re
import time
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass
import pickle
from typing import Optional, Union, Dict, List, Any
from bs4 import BeautifulSoup, Tag
import requests
from urllib.parse import urljoin
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


# data definitions


@dataclass(frozen=True)
class Page:
    """Represents a webpage from Racket documentation.
    """
    parent: Optional["Page"]
    title: str
    # url without hash
    url: str


@dataclass(frozen=True)
class Section:
    """Represents a section, subsection, subsubsection, etc. of a page of racket documentation
    """
    parent: Optional["Section"]
    title: str
    # the page that contains this section
    page: Page


# Represents a search result
Document = Section

# Represents a token of text, like a word. A token is a string that matches r'[a-z0-9]+'
Token = str
# Represents a stemmed token. Just strips suffixes.
SToken = str
# Represents a search term
Term = SToken


@dataclass(unsafe_hash=True)
class TermData:
    """Represents a term's indexed data.
    """
    # maps each document to the frequency of this term in it
    docToFrequency: Dict[Document, int]
    # the frequency of this term in the entire corpus
    totalFrequency: int


@dataclass(frozen=True)
class DocumentData:
    """Represents a document's indexed data.
    """
    # The number of terms in this document. Counts repetitions of a term separately.
    # Document length by terms.
    num_terms: int


@dataclass(frozen=True)
class Index:
    """Represents the search engine index. This is the processed form of all documents in the corpus and contains
    sufficient information to judge relevance based on a query"""
    # TermData for each term
    term_data: Dict[Term, TermData]
    num_documents: int
    # DocumentData for each document
    document_data: Dict[Document, DocumentData]


@dataclass(frozen=True)
class ScrapeTree:
    """Intermediate representation of a scrape in progress.
    Represents a hierarchical structure of headers of a page.
    Like a section that stores its content and children instead of parent"""
    # header text
    title: str
    # paragraphs, etc.
    content_children: List[Tag]
    # nested "sections"
    scrape_children: List["ScrapeTree"]


# functionality


def save_index(index: Index, out_file):
    """Save index to file"""
    pickle.dump(index, out_file)


def load_index(in_file) -> Index:
    """Load an index created from build_and_save_index"""
    index = pickle.load(in_file)
    assert isinstance(index, Index), "loaded something that isn't a saved index"
    return index


def toc_url_to_index(toc_url: str) -> Index:
    """Build an index for a search engine for the Racket Reference.

    :param toc_url: The URL of the table of contents page of the Racket Reference
    """
    print("crawling", time.time())
    pages = crawl(toc_url)
    print("scraping", time.time())
    section_to_text = dict_unions([scrape(page) for page in pages])
    print("tokenizing and stemming", time.time())
    section_to_terms = {section: stem_tokens(tokenize(text)) for section, text in section_to_text.items()}
    print("compiling to index", time.time())
    return documents_to_index(section_to_terms)


def crawl(toc_url: str) -> List[Page]:
    """Crawl the documentation to get all pages. Result order doesn't matter.
    The main table of contents itself is excluded."""
    # the table of contents lists all the pages
    # visit them all and find parents using the "up" anchor on each page
    page_urls = toc_page_urls(toc_url)
    # this filters out section part links so we only get 1 url per page, without hashes
    # assumes each section part link also has its parent page in the toc on its own
    hashless_page_urls = [url for url in page_urls if "#" not in url]
    return urls_to_pages(hashless_page_urls, toc_url)


def toc_page_urls(toc_url: str) -> List[str]:
    """Get the links to all documentation pages from the reference table of contents."""
    # hard-coded and brittle
    soup = url_to_soup(toc_url)
    main = soup.body.find_next("div", {"class": "main"})
    # that's the first thing in the table of contents
    table = main(string="Language Model")[0].parent.parent.parent.parent.parent
    # rm bibliography and index from the end
    toc_links = table.find_all('a')[:-2]
    return [urljoin(toc_url, link['href']) for link in toc_links]


@lru_cache(maxsize=None)
def url_to_soup(url: str) -> BeautifulSoup:
    """Read the webpage and create a beautiful soup object from it."""
    resp = requests.get(url)
    reference_html = resp.text
    return BeautifulSoup(reference_html, 'html.parser')


def urls_to_pages(page_urls: List[str], toc_url: str) -> List[Page]:
    """Get Pages from their urls, connecting parents.
    toc_url should not be in the list and is assumed to be the root.
    Expect urls to be in topological order, where a page is never before its parent.
    Results are in the same order.
    Assumes section part links are not in the list.
    """
    toc_page = Page(None, url_to_title(toc_url), toc_url)
    url_to_page_memo = {toc_url: toc_page}
    for page_url in page_urls:
        page = url_to_page(page_url, url_to_page_memo)
        url_to_page_memo[page_url] = page
    return [url_to_page_memo[page_url] for page_url in page_urls]


def url_to_title(page_url: str) -> str:
    """Get the title of the webpage at the given URL"""
    soup = url_to_soup(page_url)
    raw_title = soup.find('title').text
    return strip_title(raw_title)


def strip_title(raw_title: str) -> str:
    """Strip a page/section number from the beginning of a title.
    >>>strip_title("1.1 Evaluation Model")
    "Evaluation Model"
    >>>strip_title("The Racket Reference")
    "The Racket Reference"
    """
    match = re.match(r'[0-9\s.]*(.+)', raw_title)
    if not match:
        raise ValueError(f"Not a title: {raw_title}")
    return match.groups()[0]


def url_to_page(url: str, url_to_page_memo: Dict[str, Page]) -> Page:
    """Get a Page from the given URL. Use memo instead of recursive calls."""
    title = url_to_title(url)
    parent_url = get_parent_url(url)
    parent = url_to_page_memo[parent_url]
    return Page(parent, title, url)


def get_parent_url(url: str) -> str:
    """Get the url of the parent of this page of documentation.
    Assumes the webpage has one anchor with text "up" that links to its parent."""
    soup = url_to_soup(url)
    href = soup.find('a', string='up')['href']
    return urljoin(url, href)


def dict_unions(dicts: List[Dict]) -> Dict:
    """Union all dictionaries in the list. Later dicts overwrite in the case of duplicate keys."""
    ans = dict()
    for d in dicts:
        ans = {**ans, **d}
    return ans


def scrape(page: Page) -> Dict[Section, str]:
    """Scrape the sections from a page of documentation.
    Although sections are nested, there should be no overlap in text between them.
    Section titles are included in a section's text.
    """
    soup = url_to_soup(page.url)
    main = soup.find('div', class_="main")
    children = list(main.children)
    # filter out divs like the navbar and hidden secref notes
    # assumes there are no main-level divs with actual content
    children = [child for child in children if child.name != 'div']
    scrape_tree = parse_scrape_tree(children)
    return scrape_tree_to_sections(scrape_tree, page)


def parse_scrape_tree(tag_stream: List[Tag]) -> ScrapeTree:
    """Parse the main content tags (flat) to recover the nested
    section structure via header levels"""
    tag_stream = tag_stream[:]
    trees = parse_scrape_trees(tag_stream, parent_level=-1)
    assert len(trees) > 0, "The page's content did not start with a header"
    assert len(tag_stream) == 0, "Parsing didn't get all tags somehow"
    return trees[0]


def parse_scrape_trees(tag_stream: List[Tag], parent_level: int) -> List[ScrapeTree]:
    """Parse 0 or more scrape trees with header level > parent_level.
    EFFECT: Eats the beginning of tag_stream until we can't parse anymore."""
    children = []
    while len(tag_stream) != 0:
        tag = tag_stream[0]
        n = header_number(tag)
        assert n is not None
        if n <= parent_level:
            # lower n means mare parental
            break
        title = strip_title(tag.text)
        tag_stream.pop(0)
        content_children = parse_scrape_content(tag_stream)
        scrape_children = parse_scrape_trees(tag_stream, parent_level=n)
        children.append(ScrapeTree(title, content_children, scrape_children))
    return children


def parse_scrape_content(tag_stream: List[Tag]) -> List[Tag]:
    """Parse 0 or more non-header tags.
    EFFECT: Eats the beginning of tag_steam until we can't parse anymore."""
    children = []
    while len(tag_stream) != 0:
        tag = tag_stream[0]
        n = header_number(tag)
        if n is None:
            children.append(tag)
            tag_stream.pop(0)
        else:
            break
    return children


def header_number(tag: Tag) -> Optional[int]:
    """For a tag like h5, return 5. For a non-h tag, return None"""
    name = tag.name
    if len(name) <= 1:
        return None
    elif name[0] != 'h':
        # assume H isn't valid
        return None
    else:
        return int(name[1:])


def scrape_tree_to_sections(scrape_tree: ScrapeTree, page: Page) -> Dict[Section, str]:
    """Convert a scrape tree to a mapping from sections to their text.
    Prepends section titles to their body text."""
    ans = dict()
    scrape_tree_to_sections_help(scrape_tree, page, ans)
    return ans


def scrape_tree_to_sections_help(scrape_tree: ScrapeTree, page: Page, ans: Dict[Section, str], parent: Section = None):
    """Map each section to its text in ans.
    EFFECT: Adds entries to ans"""
    this_section = Section(parent, scrape_tree.title, page)
    # add space to ensure tokens don't get merged
    this_text = scrape_tree.title + ' ' + tags_to_text(scrape_tree.content_children)
    ans[this_section] = this_text
    for child_tree in scrape_tree.scrape_children:
        scrape_tree_to_sections_help(child_tree, page, ans, this_section)


def tags_to_text(tags: List[Tag]) -> str:
    """Get the text of all tags in order. Inserts spaces between."""
    return " ".join([tag.text for tag in tags])


def tokenize(text: str) -> List[Token]:
    """Tokenize text."""
    return word_tokenize(text)


def stem_tokens(tokens: List[Token]) -> List[SToken]:
    """Stem each token."""
    return [stem_token(token) for token in tokens]


stemmer = PorterStemmer()


def stem_token(token: Token) -> SToken:
    """Stem the token. Just strips suffixes. (porter stemmer)"""
    return stemmer.stem(token)


def documents_to_index(document_to_terms: Dict[Document, List[Term]]) -> Index:
    """Create an index from the documents and their corresponding terms."""
    num_documents = len(document_to_terms)
    document_data = {document: DocumentData(len(terms)) for document, terms in document_to_terms.items()}
    term_data = compute_term_data(document_to_terms)
    return Index(term_data, num_documents, document_data)


def compute_term_data(document_to_terms: Dict[Document, List[Term]]) -> Dict[Term, TermData]:
    """Compute term frequency statistics for each term"""
    ans = defaultdict(lambda: TermData(defaultdict(int), 0))
    for document, terms in document_to_terms.items():
        for term, freq in get_counts(terms).items():
            ans[term].totalFrequency += freq
            ans[term].docToFrequency[document] = freq
    for term, data in ans.items():
        data.docToFrequency = dict(data.docToFrequency)
    return dict(ans)


def get_counts(items) -> Dict[Any, int]:
    """Convert collection to a mapping from each element to its frequency in the collection"""
    ans = defaultdict(int)
    for item in items:
        ans[item] += 1
    return dict(ans)


if __name__ == '__main__':
    print("starting", time.time())
    index = toc_url_to_index("https://docs.racket-lang.org/reference/index.html")
    with open('out/index.pickle', 'wb') as out_file:
        print("saving index", time.time())
        save_index(index, out_file)
    print("done", time.time())
