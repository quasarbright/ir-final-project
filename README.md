# Mike Delmonaco ir-final-project

This repository contains the source code for my final project for a course on Information Retrieval.

For my project, I implemented the document processing component of a full-text ad-hoc search engine for racket 
documentation.

## Dependencies

* Python version 3.7.6 or greater
* Python packages listed in `requirements.txt`
* nltk punkt models. To install: https://www.nltk.org/data.html

## How to use it

Once dependencies are installed, run `main.py` to build the index. It saves it to `out/index.pickle`.
To use a saved index for a search engine, load it from file with `load_index`.