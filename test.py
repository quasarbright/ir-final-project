import unittest
import main
from bs4 import BeautifulSoup, Tag

toc_url = "https://docs.racket-lang.org/reference/index.html"


def tag(text: str) -> Tag:
    return BeautifulSoup(text, 'html.parser').find()


class CrawlTest(unittest.TestCase):
    def test_crawl(self):
        pages = main.crawl(toc_url)
        self.assertEqual(len(pages), 222)
        evaluation = find(pages, lambda page: page.title == "Evaluation Model")
        language = find(pages, lambda page: page.title == "Language Model")
        self.assertEqual(evaluation.parent, language)
        self.assertEqual(language.parent.title, "The Racket Reference")
        self.assertIsNone(language.parent.parent)
        # no duplicate urls
        urls = {page.url for page in pages}
        self.assertEqual(len(urls), len(pages))
        # no hashes in urls
        for page in pages:
            self.assertNotIn("#", page.url)

    def test_strip_title(self):
        self.assertEqual(main.strip_title("The Racket Reference"), "The Racket Reference")
        self.assertEqual(main.strip_title("1.1 Evaluation Model"), "Evaluation Model")
        self.assertEqual(main.strip_title("1.1\xa0Evaluation Model"), "Evaluation Model")


h4 = tag('<h4>t4</h4>')
h5 = tag('<h5>t5</h5>')
h6 = tag('<h6>t6</h6>')
p = tag('<p>para</p>')
bogusPage = main.Page(None, "Bogus", "bog.us")


class ScrapeTest(unittest.TestCase):
    def test_parse_scrape_tree(self):
        self.assertEqual(main.parse_scrape_tree([h4, p]), main.ScrapeTree("t4", [p], []))
        self.assertEqual(main.parse_scrape_tree([h4, p, h5, p, p]),
                         main.ScrapeTree('t4', [p], [main.ScrapeTree('t5', [p, p], [])]))
        self.assertEqual(main.parse_scrape_tree([h4, p, h5, p, p, h5, p]),
                         main.ScrapeTree('t4', [p], [main.ScrapeTree('t5', [p, p], []),
                                                     main.ScrapeTree('t5', [p], [])]))
        # TODO test h6

    def test_scrape_tree_to_sections(self):
        tree = main.parse_scrape_tree([h4, p, h5, p, p])
        section_dict = main.scrape_tree_to_sections(tree, bogusPage)
        sec4 = main.Section(None, "t4", bogusPage)
        sec5 = main.Section(sec4, "t5", bogusPage)
        self.assertEqual(section_dict,
                         {sec4: "t4 para",
                          sec5: "t5 para para"})


class ToIndexTest(unittest.TestCase):
    def test_compute_term_data(self):
        self.assertEqual(main.compute_term_data({"foo": ["a", "b", "a"], "bar": ["b", "c"]}),
                         {"a": main.TermData({"foo": 2}, 2),
                          "b": main.TermData({"foo": 1, "bar": 1}, 2),
                          "c": main.TermData({"bar": 1}, 1)})


def find(col, pred):
    """find the first element of col that returns a truthy value when applied to pred"""
    matches = [ele for ele in col if pred(ele)]
    if len(matches) == 0:
        raise ValueError("Couldn't find a value that matches the predicate")
    else:
        return matches[0]