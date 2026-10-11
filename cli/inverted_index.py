import json
import pickle
from pathlib import Path

from nltk import PorterStemmer
from nltk.tokenize.sonority_sequencing import punctuation
from rich.progress import Progress


class InvertedIndex:
    def __init__(self) -> None:
        self.index_path: Path = Path("cache/index.pkl")
        self.docmap_path: Path = Path("cache/docmap.pkl")
        self.index: dict = {}
        self.docmap: dict = {}

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

    def get_documents(self, term) -> list[int]:
        with open(self.index_path, "rb") as file:
            data = pickle.load(file)
        return sorted(data[term])

    def build(self):
        data = self.load_movies()
        with Progress() as progress:
            task = progress.add_task("Building the index: ", total=len(data["movies"]))
            for doc in data["movies"]:
                text = f"{doc['title']} {doc['description']}"
                self.__add_document(doc["id"], text)
                self.docmap[doc["id"]] = text
                progress.update(task, advance=len(data["movies"]) * 0.0005)
        self.save()

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.docmap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "+wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "+wb") as file:
            pickle.dump(self.docmap, file)

    def load_movies(self):
        with open("data/movies.json") as file:
            data = json.load(file)
        return data


if __name__ == "__main__":
    ii = InvertedIndex()
    print(ii.get_documents("hello"))
