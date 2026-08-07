"""Example hooks plugin for the ``declarative_hooks`` workflow.

Registers one named hook, ``tick``, that prints every node start/end.
Hooks are referenced by name from the workflow's ``hooks:`` block; loading
the plugin (declared under ``plugins:``) is what makes the name resolvable.

Hook signatures follow ``graph.run(hooks=...)``::

    on_node_start -> fn(node_id, node, state)
    on_node_end   -> fn(node_id, node, state, result)
    on_node_error -> fn(node_id, node, state, error)

The generic ``*args/**kwargs`` form works for all three; extra positional
args arrive for the ``on_node_end``/``on_node_error`` events.
"""

from teff import hooks


@hooks.hook("tick")
def tick(*args, **kwargs):
    node_id = args[0]
    node = args[1]
    event = "start" if len(args) == 3 else "end/error"
    print(f"[hook] {node_id} ({node.type}) {event}")
