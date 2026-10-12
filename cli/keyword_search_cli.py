import argparse
from ast import arguments
from string import punctuation

from inverted_index import InvertedIndex
from nltk.stem import PorterStemmer


def processing(text: str) -> list[str]:
    stemmer = PorterStemmer()
    with open("data/stopwords.txt") as file:
        stopwords = file.read().splitlines()
    rp = {ord(char): None for char in punctuation}
    result = list(
        filter(lambda term: term not in stopwords, text.lower().translate(rp).split())
    )
    return [stemmer.stem(term) for term in result]


def matching(query: str, title: str) -> bool:
    for q_term in processing(query):
        for t_term in processing(title):
            if q_term in t_term:
                return True
    return False


inverted_index = InvertedIndex()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="build the inverted index")
    search_parser = subparsers.add_parser("tf", help="get term frequency")
    search_parser.add_argument("doc_id", type=int, help="document id")
    search_parser.add_argument("term", type=str, help="term")
    args = parser.parse_args()

    match args.command:
        case "search":
            inverted_index.load()
            results = []
            for term in processing(args.query):
                if len(results) >= 5:
                    break
                results += inverted_index.get_documents(term)
            print(f"Searching for: {args.query}")
            for movie_id in results[:5]:
                print(f"{movie_id} {inverted_index.docmap[movie_id].split(':')[0]}")
        case "build":
            inverted_index.build()
        case "tf":
            inverted_index.load()
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print(tf)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
