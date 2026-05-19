"""
PromptEval — golden-string and live-model evaluation harness.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions

from .base import LibraryPrompt

logger = logging.getLogger(__name__)

GOLDENS_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "prompts" / "goldens"


@dataclass
class EvalResult:
    name: str
    passed: bool
    kind: str  # 'golden' | 'live'
    message: str = ""
    rendered: str = ""
    model_output: str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    def passed(self) -> list[EvalResult]:
        return [r for r in self.results if r.passed]

    def failed(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    def as_table(self) -> str:
        if not self.results:
            return "No prompts evaluated — registry is empty.\n"
        lines = [f"{'Name':<45} {'Kind':<8} {'Status'}"]
        lines.append("-" * 70)
        for r in self.results:
            status = "PASS" if r.passed else f"FAIL: {r.message}"
            lines.append(f"{r.name:<45} {r.kind:<8} {status}")
        lines.append("-" * 70)
        n_pass = len(self.passed())
        n_fail = len(self.failed())
        lines.append(f"Total: {len(self.results)}  Pass: {n_pass}  Fail: {n_fail}")
        return "\n".join(lines) + "\n"


class PromptEval:
    """Evaluation harness for LibraryPrompt instances."""

    def __init__(self, goldens_dir: Path | None = None) -> None:
        self.goldens_dir = goldens_dir or GOLDENS_DIR
        self.goldens_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Golden evaluation
    # ------------------------------------------------------------------

    def eval_golden(self, prompt: LibraryPrompt) -> EvalResult:
        """Render with fixture; compare against stored golden string."""
        try:
            rendered = prompt.render_fixture()
        except Exception as exc:
            return EvalResult(
                name=prompt.name,
                passed=False,
                kind="golden",
                message=f"render error: {exc}",
            )

        golden_path = self.goldens_dir / f"{prompt.name}.txt"

        if not golden_path.exists():
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(rendered)
            logger.warning(
                "Golden file missing for '%s'; written to %s. "
                "Review it and commit.",
                prompt.name,
                golden_path,
            )
            return EvalResult(
                name=prompt.name,
                passed=True,
                kind="golden",
                message="(golden created on first run)",
                rendered=rendered,
            )

        expected = golden_path.read_text()
        if rendered.strip() == expected.strip():
            return EvalResult(
                name=prompt.name, passed=True, kind="golden", rendered=rendered
            )
        return EvalResult(
            name=prompt.name,
            passed=False,
            kind="golden",
            message="rendered output differs from golden",
            rendered=rendered,
        )

    # ------------------------------------------------------------------
    # Live evaluation
    # ------------------------------------------------------------------

    def eval_live(self, prompt: LibraryPrompt, model: str) -> EvalResult:
        """Run rendered prompt through a model; validate against expected_shape."""
        try:
            rendered = prompt.render_fixture()
        except Exception as exc:
            return EvalResult(
                name=prompt.name,
                passed=False,
                kind="live",
                message=f"render error: {exc}",
            )

        try:
            output = self._run_model(rendered, model)
        except Exception as exc:
            return EvalResult(
                name=prompt.name,
                passed=False,
                kind="live",
                message=f"model error: {exc}",
            )

        if prompt.expected_shape is None:
            return EvalResult(
                name=prompt.name,
                passed=True,
                kind="live",
                rendered=rendered,
                model_output=output,
                message="(no shape check defined)",
            )

        ok, msg = self._check_shape(output, prompt.expected_shape)
        return EvalResult(
            name=prompt.name,
            passed=ok,
            kind="live",
            message=msg,
            rendered=rendered,
            model_output=output,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def eval_all_golden(self, prompts: list[LibraryPrompt]) -> EvalReport:
        report = EvalReport()
        for p in prompts:
            report.results.append(self.eval_golden(p))
        return report

    def eval_all_live(
        self, prompts: list[LibraryPrompt], model: str, delay: float = 2.0
    ) -> EvalReport:
        import time

        report = EvalReport()
        for i, p in enumerate(prompts):
            if i > 0 and delay > 0:
                time.sleep(delay)
            report.results.append(self.eval_live(p, model))
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_model(self, prompt_text: str, model: str) -> str:
        """Call a model via effgen's model loader and return the text output.

        Routing rules (applied in order):
        1. ``provider:model_id`` prefix is handled natively by ``load_model``.
        2. A bare model id that lives in exactly one provider catalog is
           auto-routed to that provider. (Ambiguous ids raise as usual.)
        3. Otherwise the loader's default detection runs (OpenAI/Anthropic/
           Gemini/HF).
        """
        from effgen.models import load_model

        provider = None
        if isinstance(model, str) and ":" not in model:
            try:
                # Ensure adapters are imported so the registry is populated.
                import effgen.models.cerebras_adapter  # noqa: F401
                from effgen.models.registry import ProviderRegistry

                candidates = ProviderRegistry._model_index.get(model, [])
                if len(candidates) == 1:
                    provider = candidates[0]
            except Exception:
                provider = None

        loaded = load_model(model, provider=provider) if provider else load_model(model)
        result = loaded.generate(prompt_text)
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text") or result.get("content")
        if text is None:
            text = str(result)
        return text

    @staticmethod
    def _check_shape(output: str, spec: dict[str, Any]) -> tuple[bool, str]:
        """Validate model output against an expected_shape spec."""
        kind = spec.get("type", "")

        if kind == "json":
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                # Try to extract JSON block from markdown.
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        return False, "output is not valid JSON"
                else:
                    return False, "output is not valid JSON"
            if not isinstance(parsed, dict):
                return False, "JSON output must be an object"

            schema = spec.get("schema")
            if schema:
                try:
                    Draft202012Validator.check_schema(schema)
                    Draft202012Validator(schema).validate(parsed)
                except jsonschema_exceptions.SchemaError as exc:
                    return False, f"invalid JSON schema: {exc.message}"
                except jsonschema_exceptions.ValidationError as exc:
                    return False, f"JSON schema validation failed: {exc.message}"
            return True, ""

        if kind == "regex":
            pattern = spec.get("pattern", "")
            if re.search(pattern, output, re.IGNORECASE | re.DOTALL):
                return True, ""
            return False, f"output did not match pattern: {pattern!r}"

        if kind == "callable":
            fn = spec.get("fn")
            if fn is None:
                return True, "(no callable provided)"
            try:
                result = fn(output)
                if result is True or result is None:
                    return True, ""
                return False, str(result)
            except Exception as exc:
                return False, f"shape-check raised: {exc}"

        return True, f"(unknown shape type '{kind}'; skipped)"
