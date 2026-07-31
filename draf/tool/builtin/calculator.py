"""Calculator tool — AST-based safe evaluation of math expressions."""

import ast
import operator

from draf.tool.tool import Tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool(Tool):
    """Evaluate mathematical expressions using AST-based safe eval."""

    name = "calculator"
    description = "Evaluate mathematical expressions"

    def run(self, expression: str = "") -> str:  # type: ignore[override]
        tree = ast.parse(expression, mode="eval")
        return str(self._eval(tree.body))

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.Name) and node.id == "pi":
            import math

            return math.pi
        raise ValueError(f"unsupported: {ast.dump(node)}")
