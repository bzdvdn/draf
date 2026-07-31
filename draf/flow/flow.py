"""Fluid flow builder for constructing graphs."""

from draf.node.node import Node
from draf.graph import Graph, Edge
from draf.flow.case import Case
from draf.flow.sub_flow import SubFlow
from draf.node.parallel import Parallel


class Flow:
    """Fluid builder for constructing graphs with branching.

    Args:
        name: Optional flow name.

    Usage::

        flow = Flow("my-flow")
        flow.step(LLM(model="gpt-4"))
        flow.branch("status", Case("ok").add(ok_node), default=err_node)
        graph = flow.compile()
    """

    def __init__(self, name: str = ""):
        self._name = name
        self._nodes: list[Node] = []
        self._node_ids: list[str] = []
        self._edges: list[Edge] = []
        self._counter = 0
        self._last_added: str | None = None
        self._last_branch_key: str | None = None
        self._last_branch_values: list[str] = []
        self._branch_point: str | None = None
        self._branch_ends: list[str] = []

    def _next_id(self, node: Node) -> str:
        self._counter += 1
        return f"{node.type}_{self._counter}"

    def step(self, node: Node) -> "Flow":
        """Append a node to the linear chain.

        Accepts a Node instance::

            flow.step(Transform(action="uppercase"))
            flow.step(LLM(model="gpt-4"))
            flow.step(custom_node)

        Returns ``self`` for chaining.
        """
        if not isinstance(node, Node):
            raise TypeError("step expects a Node instance")
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
        Pass *max_iterations* to limit internal steps (see :class:`SubFlow`).
        """
        sub = SubFlow(graph=flow.compile(), **kw)
        self._nodes.append(sub)
        nid = self._next_id(sub)
        self._node_ids.append(nid)
        if self._last_added is not None:
            self._edges.append(Edge(source_id=self._last_added, target_id=nid))
        self._last_added = nid
        return self

    def parallel(self, *branches) -> "Flow":
        """Run several branch chains concurrently from the last node.

        Each *branch* is a single :class:`Node`, a list of nodes (run
        sequentially inside the branch), or a :class:`Flow` (embedded as
        a :class:`SubFlow`).  Branches execute via ``asyncio.gather`` on
        isolated copies of the state; per-key reducers (``append`` etc.)
        merge their updates back without overwriting one another.

        Combine with ``converge()`` to rejoin the parallel paths::

            flow.parallel(
                [Transform(action="uppercase", input_key="a", output_key="a")],
                [Transform(action="uppercase", input_key="b", output_key="b")],
            ).converge(shout_node)
        """
        branch_specs: list[Node | list[Node]] = [self._as_branch(b) for b in branches]
        node = Parallel(branch_specs)
        self._nodes.append(node)
        nid = self._next_id(node)
        self._node_ids.append(nid)
        if self._last_added is not None:
            self._edges.append(Edge(source_id=self._last_added, target_id=nid))
        self._last_added = nid
        self._branch_ends = [nid]
        return self

    def _as_branch(self, branch) -> list[Node]:
        """Normalise a branch spec into a list of nodes."""
        if isinstance(branch, Node):
            return [branch]
        if isinstance(branch, Flow):
            return [SubFlow(graph=branch.compile())]
        return list(branch)

    def branch(self, key: str, *cases: "Case", default: Node | None = None) -> "Flow":
        """Add conditional branching from the last added node.

        Args:
            key: State key to evaluate.
            *cases: One or more Case objects, each with a value.
            default: Optional fallback node (catches unmatched values).

        Each case creates an edge ``key=<case.value>`` from the last node.
        The default creates an edge ``key!=<all case values>``.

        Multiple nodes within a case are chained sequentially.
        Use ``converge()`` after branching to merge branches.
        """
        if not cases:
            raise ValueError("branch requires at least one Case")
        assert self._last_added is not None
        self._last_branch_key = key
        self._last_branch_values = []
        self._branch_point = self._last_added
        self._branch_ends = []
        for case in cases:
            self._last_branch_values.append(case.value)
            prev_id: str | None = None
            for n in case._nodes:
                self._nodes.append(n)
                nid = self._next_id(n)
                self._node_ids.append(nid)
                parent = prev_id if prev_id is not None else self._last_added
                condition = f"{key}={case.value}" if prev_id is None else None
                self._edges.append(
                    Edge(source_id=parent, target_id=nid, condition=condition)
                )
                prev_id = nid
            if prev_id is not None:
                self._branch_ends.append(prev_id)
        if default:
            self._nodes.append(default)
            dnid = self._next_id(default)
            self._node_ids.append(dnid)
            negated = ",".join(self._last_branch_values)
            self._edges.append(
                Edge(
                    source_id=self._last_added,
                    target_id=dnid,
                    condition=f"{key}!={negated}",
                )
            )
            self._branch_ends.append(dnid)
        if self._branch_ends:
            self._last_added = self._branch_ends[-1]
        return self

    def default(self, node: Node) -> "Flow":
        """Add a fallback node for the most recent branch."""
        self._nodes.append(node)
        dnid = self._next_id(node)
        self._node_ids.append(dnid)
        bp = self._branch_point or self._last_added
        if self._last_branch_key and self._last_branch_values:
            assert bp is not None
            key = self._last_branch_key
            negated = ",".join(self._last_branch_values)
            self._edges.append(
                Edge(
                    source_id=bp,
                    target_id=dnid,
                    condition=f"{key}!={negated}",
                )
            )
        self._branch_ends.append(dnid)
        self._last_added = dnid
        return self

    def converge(self, node: Node) -> "Flow":
        """Merge all branch ends into a single node.

        Adds edges from every branch end (set by the last ``branch()``
        call) to *node*.  Use after ``branch()`` to rejoin paths::

            flow.branch("sentiment",
                Case("positive").add(on_pos),
                Case("negative").add(on_neg),
            ).converge(shout_node)
        """
        self._nodes.append(node)
        nid = self._next_id(node)
        self._node_ids.append(nid)
        for src in self._branch_ends:
            self._edges.append(Edge(source_id=src, target_id=nid))
        self._last_added = nid
        self._branch_ends = []
        return self

    def react(
        self,
        model: str,
        system: str = "",
        *,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: str = "messages",
        **config,
    ) -> "Flow":
        """Build a ReAct agent loop (LLM ↔ tool) inside this flow.

        Creates two nodes: an LLM agent and a tool executor, wired in a
        cycle.  The agent calls the LLM; if the LLM requests a tool, the
        signal ``_tool_call_name`` is set and execution passes to the
        tool executor, which runs the tool and loops back to the agent.
        When the LLM responds without a tool call, the output is stored
        at *output_key* and the graph terminates.

        Args:
            model: LLM model name (e.g. ``gpt-4``).
            system: Optional system prompt.
            input_key: State key for user input (default ``"input"``).
            output_key: State key for final response (default ``"output"``).
            messages_key: State key for conversation (default ``"messages"``).
            **config: Extra kwargs passed to :class:`ReActAgent` config
                (temperature, max_tokens, response_format, provider, etc.).

        Remember to pass ``max_iterations`` to ``graph.run()``::

            result = await graph.run(state, tools=tools, max_iterations=20)
        """
        from draf.node.agent import ReActAgent, ToolExec

        agent_cfg = {
            "model": model,
            "system": system,
            "input_key": input_key,
            "output_key": output_key,
            "messages_key": messages_key,
            **config,
        }
        agent = ReActAgent(**agent_cfg)

        self._nodes.append(agent)
        agent_id = self._next_id(agent)
        self._node_ids.append(agent_id)

        tool_exec = ToolExec(messages_key=messages_key)
        self._nodes.append(tool_exec)
        tool_id = self._next_id(tool_exec)
        self._node_ids.append(tool_id)

        self._edges.append(Edge(agent_id, tool_id, "_tool_call_name!="))
        self._edges.append(Edge(tool_id, agent_id))

        if self._last_added is not None:
            self._edges.append(Edge(self._last_added, agent_id))

        self._last_added = agent_id
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
