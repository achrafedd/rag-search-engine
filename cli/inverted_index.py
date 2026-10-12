import json
import pickle
from collections import Counter
from pathlib import Path

from nltk import PorterStemmer
from nltk.tokenize.sonority_sequencing import punctuation


class InvertedIndex:
    def __init__(self) -> None:
        self.parent: Path = Path("cache/")
        self.index_path: Path = self.parent / "index.pkl"
        self.docmap_path: Path = self.parent / "docmap.pkl"
        self.term_paths: Path = self.parent / "term_frequencies.pkl"
        self.index: dict = {}
        self.docmap: dict = {}
        self.term_frequencies: dict = {}

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id].get(term, 0)

    def tokenizer(self, term):
        token = self.processing(term)
        if 0 > len(token) > 1:
            raise ValueError("Invalid Token")
        return token[0]

    @staticmethod
    def processing(text: str) -> list[str]:
        stemmer = PorterStemmer()
        with open("data/stopwords.txt") as file:
            stopwords = file.read().splitlines()
        rp = {ord(char): None for char in punctuation}
        result = list(
            filter(
                lambda term: term not in stopwords, text.lower().translate(rp).split()
            )
        )
        return [stemmer.stem(term) for term in result]

    def __add_document(self, doc_id, text) -> None:
        tokens = self.processing(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies.setdefault(doc_id, Counter())[token] += 1

    def get_documents(self, term) -> list[int]:
        with open(self.index_path, "rb") as file:
            data = pickle.load(file)
        return sorted(data.get(term, []))

    def build(self):
        data = self.load_movies()
        for doc in data["movies"]:
            text = f"{doc['title']}:{doc['description']}"
            self.__add_document(doc["id"], text)
            self.docmap[doc["id"]] = text
        self.save()

    def save(self):
        self.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "+wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "+wb") as file:
            pickle.dump(self.docmap, file)
        with open(self.term_paths, "+wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load_movies(self):
        with open("data/movies.json") as file:
            data = json.load(file)
        return data

    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_paths, "rb") as file:
            self.term_frequencies = pickle.load(file)


if __name__ == "__main__":
    ii = InvertedIndex()
    print(ii.get_documents("hello"))
