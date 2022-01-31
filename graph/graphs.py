from typing import Optional
from collections.abc import Container, Collection


class Vertex:
    def __init__(self, label: str):
        self.label: str = label

    def __repr__(self) -> str:
        return self.label


def make_vertices(*names: tuple[str]) -> tuple[Vertex]:
    return tuple(Vertex(name) for name in names)


class Edge(Container):
    def __init__(self, vertex_1: Vertex, vertex_2: Vertex):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2

    def __eq__(self, other):
        if isinstance(other, Container):
            if (self.vertex_1 in other) and (self.vertex_2 in other):
                return True

        return False

    def __contains__(self, vertex) -> bool:
        return (vertex == self.vertex_1) or (vertex == self.vertex_2)

    def __repr__(self):
        return f'({self.vertex_1}, {self.vertex_2})'


class Graph:
    def __init__(self, vertices: Collection[Vertex], edges: Collection[Edge]):
        self.vertices = list(vertices)
        self.edges = list(edges)

        if not self.check_validity():
            raise RuntimeError('Graph not valid')

    def __repr__(self):
        repr_str = '({'

        for v in self.vertices[:-1]:
            repr_str += f'{v},'

        repr_str += f'{self.vertices[-1]}' + '},{'

        for e in self.edges[:-1]:
            repr_str += f'{e},'

        repr_str += f'{self.edges[-1]}' + '})'

        return repr_str

    def check_validity(self) -> bool:
        # Check if edges contain valid make_vertices
        for edge in self.edges:
            if edge.vertex_1 not in self.vertices or edge.vertex_2 not in self.vertices:
                print(f'Edge {edge} contains vertex not in vertices list.')
                return False

        return True


class Tree(Graph):
    def __init__(self, vertices: Collection[Vertex], edges: Collection[Edge], root: Vertex):
        self.root = root
        super().__init__(vertices, edges)

    def check_validity(self) -> bool:
        return (self.root in self.vertices) and super().check_validity()

    def __repr__(self):
        repr_str = '({'

        for v in self.vertices[:-1]:
            repr_str += f'{v},'

        repr_str += f'{self.vertices[-1]}' + '},{'

        for e in self.edges[:-1]:
            repr_str += f'{e},'

        repr_str += f'{self.edges[-1]}' + '},' + f'{self.root})'

        return repr_str


if __name__ == '__main__':
    verts = make_vertices('v0', 'v1', 'v2', 'v3', 'v4')
    e01 = Edge(verts[0], verts[1])
    e02 = Edge(verts[0], verts[2])
    e12 = Edge(verts[1], verts[2])

    g = Graph(verts, (e01, e02, e12))

    t = Tree(verts, (e01, e02, e12), verts[0])
