"""Fluid flow builder for constructing graphs."""

from draf.node.node import Node
from draf.node.registry import NodeRegistry, default_registry
from draf.graph import Graph, Edge
from draf.flow.case import Case
from draf.flow.sub_flow import SubFlow


class Flow:
    """Fluid builder for constructing graphs with branching.

    Args:
        name: Optional flow name.
        registry: Isolated node registry.
            If ``None`` (default) uses the global ``default_registry``
            for backward compatibility.  Pass a custom or cloned registry
            to prevent cross-contamination between flows.

    Usage::

        flow = Flow("my-flow")
        flow.step("llm_chat", model="gpt-4")
        flow.step(start_node)
        flow.branch("status", Case("ok").add(ok_node), default=err_node)
        graph = flow.compile()
    """

    def __init__(self, name: str = "", registry: NodeRegistry | None = None):
        self._name = name
        self._registry = registry or default_registry
        self._nodes: list[Node] = []
        self._node_ids: list[str] = []
        self._edges: list[Edge] = []
        self._counter = 0
        self._last_added: str | None = None
        self._last_branch_key: str | None = None
        self._last_branch_values: list[str] = []

    def _next_id(self, node: Node) -> str:
        self._counter += 1
        return f"{node.type}_{self._counter}"

    def step(self, node_or_type: Node | str, **config) -> "Flow":
        """Append a node to the linear chain.

        Accepts a Node instance or a registered type name (looked up
        from the flow's registry with optional *config* kwargs).

        Returns ``self`` for chaining::

            flow.step(A()).step(B()).step(C())
            flow.step("transform", action="uppercase")
        """
        if isinstance(node_or_type, str):
            node = self._registry.create(node_or_type, config)
        else:
            node = node_or_type
        self._nodes.append(node)
        nid = self._next_id(node)
        self._node_ids.append(nid)
        if self._last_added is not None:
            self._edges.append(Edge(source_id=self._last_added, target_id=nid))
        self._last_added = nid
        return self

    def add_flow(self, flow: "Flow", **kw) -> "Flow":
        """Embed a sub-flow as a single node (SubFlow).

        The *flow* is compiled and wrapped in a SubFlow node.
        Pass *input_map* / *output_map* as keyword arguments for key remapping.
        """
        sub = SubFlow(graph=flow.compile(), **kw)
        self._nodes.append(sub)
        nid = self._next_id(sub)
        self._node_ids.append(nid)
        if self._last_added is not None:
            self._edges.append(Edge(source_id=self._last_added, target_id=nid))
        self._last_added = nid
        return self

    def branch(self, key: str, *cases: "Case", default: Node | None = None) -> "Flow":
        """Add conditional branching from the last added node.

        Args:
            key: State key to evaluate.
            *cases: One or more Case objects, each with a value.
            default: Optional fallback node (catches unmatched values).

        Each case creates an edge ``key=<case.value>`` from the last node.
        The default creates an edge ``key!=<all case values>``.
        """
        if not cases:
            raise ValueError("branch requires at least one Case")
        assert self._last_added is not None
        self._last_branch_key = key
        self._last_branch_values = []
        for case in cases:
            self._last_branch_values.append(case.value)
            for n in case._nodes:
                self._nodes.append(n)
                nid = self._next_id(n)
                self._node_ids.append(nid)
                condition = f"{key}={case.value}"
                self._edges.append(Edge(source_id=self._last_added, target_id=nid, condition=condition))
        if default:
            self._nodes.append(default)
            dnid = self._next_id(default)
            self._node_ids.append(dnid)
            negated = ",".join(self._last_branch_values)
            self._edges.append(Edge(source_id=self._last_added, target_id=dnid, condition=f"{key}!={negated}"))
        return self

    def default(self, node: Node) -> "Flow":
        """Add a fallback node for the most recent branch."""
        self._nodes.append(node)
        dnid = self._next_id(node)
        self._node_ids.append(dnid)
        if self._last_branch_key and self._last_branch_values:
            assert self._last_added is not None
            key = self._last_branch_key
            negated = ",".join(self._last_branch_values)
            self._edges.append(
                Edge(
                    source_id=self._last_added,
                    target_id=dnid,
                    condition=f"{key}!={negated}",
                )
            )
        return self

    def compile(self) -> Graph:
        """Compile the flow into a ``Graph`` ready for execution.

        Raises:
            ValueError: If no nodes were added.
        """
        if not self._nodes:
            raise ValueError("no nodes in flow")
        return Graph(
            nodes=dict(zip(self._node_ids, self._nodes)),
            edges=self._edges,
            entry_point=self._node_ids[0],
        )
