import argparse
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


def build_command() -> None:
    docs = inverted_index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Search movies using keywords")
    args = parser.parse_args()

    match args.command:
        case "search":
            results = list(
                filter(lambda it: matching(args.query, it["title"]), data["movies"])
            )
            print(f"Searching for: {args.query}")
            print(
                *[
                    f"{idx}. {ele['title']}"
                    for idx, ele in enumerate(
                        sorted(results, key=lambda x: x["id"])[:5], start=1
                    )
                ],
                end="",
                sep="\n",
            )
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
