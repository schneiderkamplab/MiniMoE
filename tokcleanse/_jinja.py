"""Internal Jinja AST helpers."""

from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, nodes

_MEDIA_TYPE_TO_LITERALS = {
    "image": frozenset({"<|image>", "<|image|>", "<image|>"}),
    "audio": frozenset({"<|audio>", "<|audio|>", "<audio|>"}),
    "video": frozenset({"<|video|>"}),
}


def _rewrite_template(
    source: str,
    *,
    replacements: dict[str, str],
    dropped_literals: frozenset[str],
) -> str:
    ast = Environment().parse(source)
    dropped_media_types = _collect_dropped_media_types(dropped_literals)
    ast.body = _rewrite_statements(ast.body, dropped_media_types)
    _rewrite_literal_nodes(ast, replacements=replacements, dropped_literals=dropped_literals)
    return _JinjaAstEmitter().emit(ast)


def _collect_dropped_media_types(dropped_literals: frozenset[str]) -> set[str]:
    return {
        media_type
        for media_type, literals in _MEDIA_TYPE_TO_LITERALS.items()
        if literals & dropped_literals
    }


def _rewrite_statements(
    statements: list[nodes.Node],
    dropped_media_types: set[str],
) -> list[nodes.Node]:
    rewritten: list[nodes.Node] = []
    for statement in statements:
        rewritten.extend(_rewrite_statement(statement, dropped_media_types))
    return rewritten


def _rewrite_statement(
    statement: nodes.Node,
    dropped_media_types: set[str],
) -> list[nodes.Node]:
    if isinstance(statement, nodes.If):
        return _rewrite_if_statement(statement, dropped_media_types)
    if isinstance(statement, nodes.For):
        statement.body = _rewrite_statements(statement.body, dropped_media_types)
        statement.else_ = _rewrite_statements(statement.else_, dropped_media_types)
    elif isinstance(statement, nodes.Macro):
        statement.body = _rewrite_statements(statement.body, dropped_media_types)
    return [statement]


def _rewrite_if_statement(
    statement: nodes.If,
    dropped_media_types: set[str],
) -> list[nodes.Node]:
    branches = [statement, *statement.elif_]
    rewritten_else = _rewrite_statements(statement.else_, dropped_media_types)
    kept_branches: list[nodes.If] = []
    for branch in branches:
        branch.body = _rewrite_statements(branch.body, dropped_media_types)
        branch.elif_ = []
        branch.else_ = []
        media_type = _extract_item_type_comparison(branch.test)
        if media_type in dropped_media_types:
            continue
        kept_branches.append(branch)

    if not kept_branches:
        return rewritten_else

    primary = kept_branches[0]
    primary.elif_ = kept_branches[1:]
    primary.else_ = rewritten_else
    return [primary]


def _extract_item_type_comparison(test: nodes.Expr) -> str | None:
    if not isinstance(test, nodes.Compare):
        return None
    if len(test.ops) != 1:
        return None
    operand = test.ops[0]
    if operand.op != "eq" or not isinstance(operand.expr, nodes.Const):
        return None
    expr = test.expr
    if not isinstance(expr, nodes.Getitem):
        return None
    if not isinstance(expr.node, nodes.Name) or expr.node.name != "item":
        return None
    if not isinstance(expr.arg, nodes.Const) or expr.arg.value != "type":
        return None
    if not isinstance(operand.expr.value, str):
        return None
    return operand.expr.value


def _rewrite_literal_nodes(
    node: nodes.Node,
    *,
    replacements: dict[str, str],
    dropped_literals: frozenset[str],
) -> None:
    if isinstance(node, nodes.TemplateData):
        node.data = _rewrite_string_literals(
            node.data,
            replacements=replacements,
            dropped_literals=dropped_literals,
        )
    elif isinstance(node, nodes.Const) and isinstance(node.value, str):
        node.value = _rewrite_string_literals(
            node.value,
            replacements=replacements,
            dropped_literals=dropped_literals,
        )

    for field in node.fields:
        value = getattr(node, field)
        if isinstance(value, nodes.Node):
            _rewrite_literal_nodes(
                value,
                replacements=replacements,
                dropped_literals=dropped_literals,
            )
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, nodes.Node):
                    _rewrite_literal_nodes(
                        item,
                        replacements=replacements,
                        dropped_literals=dropped_literals,
                    )


def _rewrite_string_literals(
    value: str,
    *,
    replacements: dict[str, str],
    dropped_literals: frozenset[str],
) -> str:
    rewritten = value
    for literal, replacement in sorted(
        replacements.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if literal == replacement:
            continue
        rewritten = rewritten.replace(literal, replacement)
    for literal in sorted(dropped_literals, key=len, reverse=True):
        rewritten = rewritten.replace(literal, "")
    return rewritten


class _JinjaAstEmitter:
    def emit(self, node: nodes.Node) -> str:
        emitter = getattr(self, f"_emit_{type(node).__name__}", None)
        if emitter is None:
            raise TypeError(f"Unsupported Jinja AST node: {type(node).__name__}")
        return emitter(node)

    def _emit_Template(self, node: nodes.Template) -> str:
        return self._emit_statements(node.body)

    def _emit_Output(self, node: nodes.Output) -> str:
        parts: list[str] = []
        for child in node.nodes:
            if isinstance(child, nodes.TemplateData):
                parts.append(child.data)
                continue
            parts.append("{{ ")
            parts.append(self.emit(child))
            parts.append(" }}")
        return self._join_chunks(parts)

    def _emit_TemplateData(self, node: nodes.TemplateData) -> str:
        return node.data

    def _emit_Assign(self, node: nodes.Assign) -> str:
        return f"{{% set {self.emit(node.target)} = {self.emit(node.node)} %}}"

    def _emit_AssignBlock(self, node: nodes.AssignBlock) -> str:
        filter_str = "" if node.filter is None else f"| {self.emit(node.filter)}"
        body = self._emit_statements(node.body)
        return f"{{% set {self.emit(node.target)}{filter_str} %}}{body}{{% endset %}}"

    def _emit_For(self, node: nodes.For) -> str:
        header = f"{{% for {self.emit(node.target)} in {self.emit(node.iter)}"
        if node.test is not None:
            header += f" if {self.emit(node.test)}"
        if node.recursive:
            header += " recursive"
        header += " %}"
        body = self._emit_statements(node.body)
        if node.else_:
            body += "{% else %}" + self._emit_statements(node.else_)
        return header + body + "{% endfor %}"

    def _emit_If(self, node: nodes.If) -> str:
        parts = [f"{{% if {self.emit(node.test)} %}}", self._emit_statements(node.body)]
        for elif_node in node.elif_:
            parts.extend(
                [
                    f"{{% elif {self.emit(elif_node.test)} %}}",
                    self._emit_statements(elif_node.body),
                ]
            )
        if node.else_:
            parts.extend(["{% else %}", self._emit_statements(node.else_)])
        parts.append("{% endif %}")
        return "".join(parts)

    def _emit_Macro(self, node: nodes.Macro) -> str:
        args: list[str] = []
        default_offset = len(node.args) - len(node.defaults)
        for index, argument in enumerate(node.args):
            rendered = self.emit(argument)
            if index >= default_offset:
                rendered += "=" + self.emit(node.defaults[index - default_offset])
            args.append(rendered)
        signature = ", ".join(args)
        return (
            f"{{% macro {node.name}({signature}) %}}"
            f"{self._emit_statements(node.body)}"
            "{% endmacro %}"
        )

    def _emit_Name(self, node: nodes.Name) -> str:
        return node.name

    def _emit_NSRef(self, node: nodes.NSRef) -> str:
        return f"{node.name}.{node.attr}"

    def _emit_Const(self, node: nodes.Const) -> str:
        value = node.value
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "none"
        return repr(value)

    def _emit_Getattr(self, node: nodes.Getattr) -> str:
        return f"{self._wrap_primary_expression(node.node)}.{node.attr}"

    def _emit_Getitem(self, node: nodes.Getitem) -> str:
        target = self._wrap_primary_expression(node.node)
        if isinstance(node.arg, nodes.Slice):
            return f"{target}[{self.emit(node.arg)}]"
        return f"{target}[{self.emit(node.arg)}]"

    def _emit_Slice(self, node: nodes.Slice) -> str:
        parts = [
            "" if node.start is None else self.emit(node.start),
            "" if node.stop is None else self.emit(node.stop),
        ]
        rendered = f"{parts[0]}:{parts[1]}"
        if node.step is not None:
            rendered += f":{self.emit(node.step)}"
        return rendered

    def _emit_Tuple(self, node: nodes.Tuple) -> str:
        items = ", ".join(self.emit(item) for item in node.items)
        if len(node.items) == 1:
            items += ","
        return f"({items})"

    def _emit_List(self, node: nodes.List) -> str:
        return "[" + ", ".join(self.emit(item) for item in node.items) + "]"

    def _emit_Keyword(self, node: nodes.Keyword) -> str:
        return f"{node.key}={self.emit(node.value)}"

    def _emit_Call(self, node: nodes.Call) -> str:
        arguments = [self.emit(argument) for argument in node.args]
        arguments.extend(self.emit(keyword) for keyword in node.kwargs)
        if node.dyn_args is not None:
            arguments.append("*" + self.emit(node.dyn_args))
        if node.dyn_kwargs is not None:
            arguments.append("**" + self.emit(node.dyn_kwargs))
        return f"{self._wrap_primary_expression(node.node)}({', '.join(arguments)})"

    def _emit_Filter(self, node: nodes.Filter) -> str:
        rendered = f"{self._wrap_filter_subject(node.node)}|{node.name}"
        arguments = [self.emit(argument) for argument in node.args]
        arguments.extend(self.emit(keyword) for keyword in node.kwargs)
        if node.dyn_args is not None:
            arguments.append("*" + self.emit(node.dyn_args))
        if node.dyn_kwargs is not None:
            arguments.append("**" + self.emit(node.dyn_kwargs))
        if arguments:
            rendered += f"({', '.join(arguments)})"
        return rendered

    def _emit_Test(self, node: nodes.Test) -> str:
        rendered = f"{self._wrap_filter_subject(node.node)} is {node.name}"
        arguments = [self.emit(argument) for argument in node.args]
        arguments.extend(self.emit(keyword) for keyword in node.kwargs)
        if node.dyn_args is not None:
            arguments.append("*" + self.emit(node.dyn_args))
        if node.dyn_kwargs is not None:
            arguments.append("**" + self.emit(node.dyn_kwargs))
        if arguments:
            rendered += f"({', '.join(arguments)})"
        return rendered

    def _emit_Compare(self, node: nodes.Compare) -> str:
        rendered = self.emit(node.expr)
        for operand in node.ops:
            rendered += f" {self._emit_comparison_operator(operand.op)} {self.emit(operand.expr)}"
        return rendered

    def _emit_Operand(self, node: nodes.Operand) -> str:
        return f"{self._emit_comparison_operator(node.op)} {self.emit(node.expr)}"

    def _emit_Add(self, node: nodes.Add) -> str:
        return f"({self.emit(node.left)} + {self.emit(node.right)})"

    def _emit_Sub(self, node: nodes.Sub) -> str:
        return f"({self.emit(node.left)} - {self.emit(node.right)})"

    def _emit_Mul(self, node: nodes.Mul) -> str:
        return f"({self.emit(node.left)} * {self.emit(node.right)})"

    def _emit_Div(self, node: nodes.Div) -> str:
        return f"({self.emit(node.left)} / {self.emit(node.right)})"

    def _emit_FloorDiv(self, node: nodes.FloorDiv) -> str:
        return f"({self.emit(node.left)} // {self.emit(node.right)})"

    def _emit_Mod(self, node: nodes.Mod) -> str:
        return f"({self.emit(node.left)} % {self.emit(node.right)})"

    def _emit_Pow(self, node: nodes.Pow) -> str:
        return f"({self.emit(node.left)} ** {self.emit(node.right)})"

    def _emit_And(self, node: nodes.And) -> str:
        return f"({self.emit(node.left)} and {self.emit(node.right)})"

    def _emit_Or(self, node: nodes.Or) -> str:
        return f"({self.emit(node.left)} or {self.emit(node.right)})"

    def _emit_Not(self, node: nodes.Not) -> str:
        return f"(not {self.emit(node.node)})"

    def _emit_Neg(self, node: nodes.Neg) -> str:
        return f"(-{self.emit(node.node)})"

    def _emit_Pos(self, node: nodes.Pos) -> str:
        return f"(+{self.emit(node.node)})"

    def _emit_CondExpr(self, node: nodes.CondExpr) -> str:
        if node.expr2 is None:
            return f"({self.emit(node.expr1)} if {self.emit(node.test)})"
        return f"({self.emit(node.expr1)} if {self.emit(node.test)} else {self.emit(node.expr2)})"

    def _emit_statements(self, statements: list[nodes.Node]) -> str:
        return self._join_chunks([self.emit(statement) for statement in statements])

    def _join_chunks(self, chunks: list[str]) -> str:
        if not chunks:
            return ""
        joined = [chunks[0]]
        for chunk in chunks[1:]:
            if joined[-1].endswith("{") and chunk.startswith("{"):
                joined[-1] = joined[-1][:-1]
                joined.append("{{ '{' }}")
            joined.append(chunk)
        return "".join(joined)

    def _wrap_primary_expression(self, node: nodes.Node) -> str:
        rendered = self.emit(node)
        if isinstance(node, (nodes.Name, nodes.NSRef, nodes.Getattr, nodes.Getitem, nodes.Call)):
            return rendered
        return f"({rendered})"

    def _wrap_filter_subject(self, node: nodes.Node) -> str:
        rendered = self.emit(node)
        if isinstance(node, (nodes.Name, nodes.NSRef, nodes.Getattr, nodes.Getitem, nodes.Call)):
            return rendered
        return f"({rendered})"

    def _emit_comparison_operator(self, operator: str) -> str:
        operators = {
            "eq": "==",
            "ne": "!=",
            "gt": ">",
            "gteq": ">=",
            "lt": "<",
            "lteq": "<=",
            "in": "in",
            "notin": "not in",
        }
        try:
            return operators[operator]
        except KeyError as exc:
            raise TypeError(f"Unsupported Jinja comparison operator: {operator}") from exc


__all__: list[str] = []
