"""RST trees as produced by DMRST, and the features read off them for a target word."""

import json
import re
from dataclasses import dataclass
from functools import cached_property

SPAN_PATTERN = re.compile(r"\((\d+):(\w+)=([\w-]+):(\d+),(\d+):(\w+)=([\w-]+):(\d+)\)")


@dataclass
class RstNode:
    start: int          # first EDU, 0-based
    split: int          # last EDU of the left child
    end: int            # last EDU
    left_nuclearity: str
    right_nuclearity: str
    relation: str

    def side_of(self, edu):
        return "left" if edu <= self.split else "right"

    def nuclearity_of(self, edu):
        return self.left_nuclearity if self.side_of(edu) == "left" else self.right_nuclearity


def parse_dmrst_spans(span_string):
    """Turn DMRST's '(1:Nucleus=span:3,4:Satellite=Elaboration:6) ...' into nodes."""
    nodes = []
    for match in SPAN_PATTERN.finditer(span_string or ""):
        start, left_nuclearity, left_relation, split, _, right_nuclearity, right_relation, end = match.groups()
        # Mononuclear relations put the label on the satellite and 'span' on the nucleus
        relation = left_relation if left_relation != "span" else right_relation
        nodes.append(RstNode(int(start) - 1, int(split) - 1, int(end) - 1,
                             left_nuclearity[0], right_nuclearity[0], relation))
    return nodes


@dataclass
class RstTree:
    number_of_edus: int
    nodes: list[RstNode]

    @cached_property
    def ancestors(self):
        """For every EDU, the nodes that contain it from the root down."""
        paths = [[] for _ in range(self.number_of_edus)]
        for node in sorted(self.nodes, key=lambda n: n.end - n.start, reverse=True):
            for edu in range(node.start, min(node.end, self.number_of_edus - 1) + 1):
                paths[edu].append(node)
        return paths

    def depth(self, edu):
        return len(self.ancestors[edu])

    def attachment(self, edu):
        """Relation and nuclearity of the EDU in its innermost node."""
        path = self.ancestors[edu]
        if not path:
            return "none", "N"
        innermost = path[-1]
        return innermost.relation, innermost.nuclearity_of(edu)

    def nucleus_share(self, edu):
        """Share of the EDU's ancestors in which it sits on the nucleus side (a salience measure)."""
        path = self.ancestors[edu]
        if not path:
            return 1.0
        return sum(node.nuclearity_of(edu) == "N" for node in path) / len(path)

    def distance(self, first_edu, second_edu):
        """Number of tree edges between two EDUs."""
        first_path, second_path = self.ancestors[first_edu], self.ancestors[second_edu]
        shared = 0
        for first_node, second_node in zip(first_path, second_path):
            if first_node is not second_node:
                break
            shared += 1
        return (len(first_path) - shared) + (len(second_path) - shared)


@dataclass
class ParsedDocument:
    tree: RstTree
    edu_of_token: dict[str, list[int]]  # sentence_id -> EDU of each token


def load_rst_documents(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as trees_file:
        raw = json.load(trees_file)
    return {
        document_id: ParsedDocument(
            RstTree(entry["number_of_edus"], parse_dmrst_spans(entry["spans"])),
            entry["edu_of_token"],
        )
        for document_id, entry in raw.items()
    }


def depth_bucket(depth):
    if depth <= 3:
        return "shallow"
    if depth <= 8:
        return "middle"
    return "deep"


def share_bucket(share):
    return "mostly_nucleus" if share >= 0.67 else "mixed" if share >= 0.34 else "mostly_satellite"


def tree_distance_bucket(distance):
    if distance <= 2:
        return "close"
    if distance <= 6:
        return "middle"
    return "far"


def rst_description(document, sentence_id, token_index):
    edu = document.edu_of_token[sentence_id][token_index]
    relation, nuclearity = document.tree.attachment(edu)
    return {
        "edu": edu,
        "rst_relation": relation,
        "rst_nuclearity": nuclearity,
        "rst_depth": document.tree.depth(edu),
        "rst_nucleus_share": document.tree.nucleus_share(edu),
    }


def rst_features(description):
    return [
        f"rst:relation={description['rst_relation']}",
        f"rst:nuclearity={description['rst_nuclearity']}",
        f"rst:relation_nuclearity={description['rst_relation']}_{description['rst_nuclearity']}",
        f"rst:depth={depth_bucket(description['rst_depth'])}",
        f"rst:salience={share_bucket(description['rst_nucleus_share'])}",
    ]
