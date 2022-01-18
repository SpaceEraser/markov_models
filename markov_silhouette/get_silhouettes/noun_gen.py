from itertools import chain, islice
from math import inf
from pprint import pprint
from nltk.corpus import wordnet as wn
import nltk

nltk.download('wordnet')


def dfs(arr_in, text):
    def dfs_impl(arr):
        best_depth = inf
        for e in arr:
            if not isinstance(e, list):
                if e.name().startswith(text):
                    return 0
            else:
                best_depth = min(dfs_impl(e), best_depth)
        return best_depth + 1

    text = text + "."
    return dfs_impl(arr_in)


def find_min_max_depths_for_arr(arr, words: list[str]) -> tuple[int, int]:
    depths = list(filter(lambda x: x != inf, (dfs(arr, w) for w in words)))
    return min(depths), max(depths)


def find_at_depth(arr, depth):
    if depth == 0:
        return (e for e in arr if not isinstance(e, list))
    return chain.from_iterable(
        find_at_depth(e, depth - 1) for e in arr if isinstance(e, list)
    )


def find_between_depths(arr: str, min_depth: int, max_depth: int):
    return chain.from_iterable(
        find_at_depth(arr, i) for i in range(min_depth, max_depth + 1)
    )


def nouns_from_synset(synset_str: str, prototypes: list[str]):
    big_tree = wn.synset(synset_str).tree(lambda s: s.hyponyms())
    min_d, max_d = find_min_max_depths_for_arr(big_tree, prototypes)
    # print(f"{min_d=} | {max_d=}")

    nouns = set(
        s.name().rsplit(".", 2)[0] for s in find_between_depths(big_tree, min_d, max_d)
    )
    return (n for n in nouns if 3 <= len(n) <= 10)


def read_frequency(filename: str) -> dict[str, int]:
    with open(filename, "r") as f:
        return {w: int(f) for [w, f] in map(lambda x: x.split(" "), f.readlines())}


def word_frequencies() -> dict[str, int]:
    return {
        w: f
        for (w, f) in read_frequency("/Users/bence/Downloads/freq_en_full.txt").items()
    }

def synsets_at_level(arr_tree: list, level: int):
    if level == 0:
        return [arr_tree[0]]

    return chain.from_iterable(synsets_at_level(a, level - 1) for a in arr_tree[1:])


def words_around_synset(synset_str: str, depth: int) -> set:
    synset_hypernyms_tree = wn.synset(synset_str).tree(
        lambda s: s.hypernyms(), depth=depth
    )
    parents = synsets_at_level(synset_hypernyms_tree, level=depth)
    parents_hyponym_trees = (
        s.tree(lambda x: x.hyponyms(), depth=depth) for s in parents
    )
    return set(
        chain.from_iterable(
            synsets_at_level(t, level=depth) for t in parents_hyponym_trees
        )
    )


def find_more_nouns_like(synset_strs: list[str]) -> set:
    return set(chain.from_iterable(words_around_synset(p, 2) for p in synset_strs))


def noun_gen():
    prototype_synset_strs = [
        "cat.n.01",
        "dog.n.01",
        "car.n.01",
        "tree.n.01",
        "bear.n.01",
        "man.n.01",
        "woman.n.01",
        "train.n.01",
        "moon.n.01",
        "cloud.n.02",
        "star.n.01",
        "planet.n.01",
        "tornado.n.01",
        "back.n.01",
        "river.n.01",
        "building.n.01",
        "puppy.n.01",
        "kitten.n.01",
        "shark.n.01",
        "bull.n.01",
        "breast.n.02",
        "passport.n.02",
        "clock.n.01",
        "lip.n.01",
        "grape.n.01",
        "telephone.n.01",
        "spectacles.n.01",
        "ice_bear.n.01",
    ]
    words = (
        s.name().rsplit(".", 2)[0] for s in find_more_nouns_like(prototype_synset_strs)
    )
    words = (w.replace("_", " ").replace("-", " ") for w in words)
    words = set(words)
    yield from words


if __name__ == "__main__":
    pprint(list(islice(noun_gen(), 30)))
