import React from 'react';
import { Link } from 'react-router-dom';
import { Zap } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Execution() {
  return (
    <DocPage
      title="Code Execution"
      subtitle="Secure sandboxed code execution with Docker support, resource limits, and security validation."
      icon={<Zap size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Execution' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        effGen provides secure code execution capabilities through sandboxed environments.
        Code can be executed locally in restricted mode or in Docker containers for full isolation.
      </p>

      <InfoBox type="success" title="New in v0.3.1 — the sandbox toggle is out of the model's hands">
        <p>
          The Python REPL&apos;s <code>restricted_mode</code> switch is no longer advertised in the
          model-facing tool schema, so a prompt-injected model can never flip it. Unrestricted
          execution is now a <strong>developer-only opt-in</strong>{' '}
          (<code>PythonREPL(allow_unrestricted=True)</code> or{' '}
          <code>EFFGEN_REPL_ALLOW_UNRESTRICTED</code>); a model-supplied{' '}
          <code>restricted_mode=False</code> is ignored and execution stays sandboxed — fail-closed.
          The <code>bash</code> tool&apos;s environment scrub now strips every provider credential
          (and anything matching <code>*_API_KEY</code> / <code>*_TOKEN</code> /{' '}
          <code>*SECRET*</code> / <code>*PASSWORD*</code>), refuses reads of common secret files, no
          longer claims to run &quot;safely&quot;, and is no longer bundled in the{' '}
          <code>general</code> preset.
        </p>
      </InfoBox>

      <FeatureList
        features={[
          { icon: '🔒', title: 'Security Validation', description: 'Pre-execution code analysis for dangerous patterns' },
          { icon: '🐳', title: 'Docker Sandboxing', description: 'Full isolation with Docker containers' },
          { icon: '⏱️', title: 'Resource Limits', description: 'CPU, memory, and time limits' },
          { icon: '🔄', title: 'Retry Logic', description: 'Automatic retry on transient failures' },
        ]}
      />

      <h2>Code Executor</h2>

      <CodeBlock
        code={`from effgen.execution import CodeExecutor, SandboxConfig
import asyncio

# Sandbox configuration goes inside SandboxConfig
config = SandboxConfig(
    timeout=30,                  # Execution timeout (seconds)
    memory_limit="512M",         # Memory limit
    allow_network=False,         # Disable network access
    allow_file_ops=False,        # Disable file operations
)

# CodeExecutor picks the sandbox; config is shared between local & docker
executor = CodeExecutor(sandbox_type="local", config=config)  # "local" | "docker"

async def run_code():
    result = await executor.execute(
        code='''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])
''',
        language="python",
    )

    print(f"Status: {result.status.value}")
    print(f"Output:\\n{result.output}")
    print(f"Execution time: {result.execution_time:.2f}s")

asyncio.run(run_code())`}
        language="python"
        filename="code_executor.py"
      />

      <h2>Sandbox Configuration</h2>

      <CodeBlock
        code={`from effgen.execution import SandboxConfig, CodeExecutor

# Local sandbox (restricted Python environment, AST validator + subprocess)
local_config = SandboxConfig(
    timeout=30,
    memory_limit="256M",
    allow_network=False,
    allow_file_ops=False,
    custom_allow_imports={"math", "json", "datetime", "collections", "itertools"},
)

# Docker sandbox — same SandboxConfig, swap sandbox_type
docker_config = SandboxConfig(
    timeout=60,
    memory_limit="1G",
    cpu_limit=1.0,             # 1 CPU core
    allow_network=False,       # network=none
    allow_file_ops=False,
    max_output_size=1024 * 1024,
)

executor = CodeExecutor(sandbox_type="docker", config=docker_config)`}
        language="python"
        filename="sandbox_config.py"
      />

      <h2>Security Validation</h2>
      <p>
        The <code>CodeValidator</code> analyzes code before execution:
      </p>

      <CodeBlock
        code={`from effgen.execution import CodeValidator

# CodeValidator inspects code for dangerous patterns BEFORE execution.
# Allow-flags are flipped to permit the corresponding operation when True;
# custom_allow_imports extends the default safe-imports allow-list.
validator = CodeValidator(
    allow_network=False,
    allow_file_ops=False,
    custom_allow_imports={"math", "json"},
)

code = """
import os
import subprocess
result = subprocess.run(['ls', '-la'], capture_output=True)
print(result.stdout)
"""

validation = validator.validate(code, language="python")

print(f"Is safe: {validation.is_safe}")
print(f"Has critical: {validation.has_critical}")
print(f"Has errors: {validation.has_errors}")
for issue in validation.issues:
    print(f"  - [{issue.severity.value}] {issue.message}")
    if issue.line_number:
        print(f"    Line {issue.line_number}: {issue.code_snippet or ''}")`}
        language="python"
        filename="code_validator.py"
      />

      <h3>Validation Severity Levels</h3>

      <ApiTable
        headers={['Severity', 'Description', 'Action']}
        rows={[
          [<code>INFO</code>, 'Informational annotation', 'Execute normally'],
          [<code>WARNING</code>, 'Suspicious but allowed', 'Execute with logging'],
          [<code>ERROR</code>, 'Dangerous pattern', 'Block — execution should not proceed'],
          [<code>CRITICAL</code>, 'Definite security violation', 'Hard block; alert if needed'],
        ]}
      />

      <h2>Execution Pool</h2>
      <p>
        Run multiple code executions in parallel with resource management:
      </p>

      <CodeBlock
        code={`from effgen.execution.sandbox import ExecutionPool, SandboxConfig

# Pool of N CodeExecutors sharing one SandboxConfig
pool = ExecutionPool(
    sandbox_type="local",
    config=SandboxConfig(timeout=30, memory_limit="512M"),
    pool_size=4,
)

snippets = ["print(1 + 1)", "print(2 * 3)", "print(10 / 2)"]
for code in snippets:
    result = pool.execute(code, language="python")
    print(result.status.value, result.output)`}
        language="python"
        filename="execution_pool.py"
      />

      <h2>Python REPL</h2>
      <p>
        Interactive Python with state persistence between calls:
      </p>

      <InfoBox type="success" title="v0.3.0 — out-of-process timeout &amp; resource caps">
        <p>
          As of v0.3.0, <code>PythonREPL</code> runs user code in a worker subprocess with a hard
          wall-clock timeout, a process-group kill, and memory / output caps enforced{' '}
          <strong>outside</strong> the executed code — so a <code>while True: pass</code> dies at
          its timeout instead of ~30 s later. It is part of the broader v0.3.0 tool hardening
          (shared SSRF guard on URL tools, path-confined file tools, no <code>pickle</code> /{' '}
          <code>eval</code>) — see <a href="/docs/security">Security</a>.
        </p>
      </InfoBox>

      <CodeBlock
        code={`from effgen.tools.builtin import PythonREPL

repl = PythonREPL(
    timeout=30,
    memory_limit="512m"
)

# State persists across calls
await repl.execute("x = 10")
await repl.execute("y = 20")
result = await repl.execute("print(x + y)")
# Output: 30

# Import and use libraries
await repl.execute("import math")
result = await repl.execute("print(math.sqrt(144))")
# Output: 12.0

# Complex multi-line code
await repl.execute('''
def calculate_stats(numbers):
    return {
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers)
    }
''')

result = await repl.execute('''
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stats = calculate_stats(data)
print(stats)
''')

# Reset state
repl.reset()`}
        language="python"
        filename="python_repl.py"
      />

      <h2>Error Handling</h2>
      <p>
        <code>CodeExecutor.execute()</code> does NOT raise on validation/timeout/resource
        failures — those surface through <code>ExecutionResult.status</code> instead. Only
        unrecoverable internal exceptions propagate.
      </p>

      <CodeBlock
        code={`from effgen.execution import ExecutionStatus

result = await executor.execute(
    code="while True: pass",   # Infinite loop, will hit timeout
    language="python",
)

if result.status == ExecutionStatus.SUCCESS:
    print("OK:", result.output)
elif result.status == ExecutionStatus.TIMEOUT:
    print(f"Timed out after {result.execution_time:.1f}s")
elif result.status == ExecutionStatus.VALIDATION_FAILED:
    print("Blocked by CodeValidator:")
    for issue in result.validation_result.issues:
        print(f"  [{issue.severity.value}] {issue.message}")
elif result.status == ExecutionStatus.RESOURCE_LIMIT_EXCEEDED:
    print(f"Resource limit exceeded: {result.error}")
else:                                   # ExecutionStatus.ERROR
    print(f"Execution error: {result.error}")
    print(f"Stderr: {result.metadata.get('stderr', '')}")`}
        language="python"
        filename="error_handling.py"
      />

      <h2>Best Practices</h2>

      <InfoBox type="warning" title="Security Guidelines">
        <ul>
          <li><strong>Always validate:</strong> Run CodeValidator before execution</li>
          <li><strong>Use Docker:</strong> For untrusted code, use Docker sandboxing</li>
          <li><strong>Set limits:</strong> Always configure timeout and memory limits</li>
          <li><strong>Disable network:</strong> Unless specifically needed, disable network access</li>
          <li><strong>Whitelist imports:</strong> Only allow necessary Python imports</li>
          <li><strong>Log everything:</strong> Keep detailed logs for security auditing</li>
        </ul>
      </InfoBox>

      <InfoBox type="success" title="Next Steps">
        <p>
          Learn about <Link to="/configuration">Configuration</Link> for deployment settings,
          or check the <Link to="/api-reference">API Reference</Link> for detailed documentation.
        </p>
      </InfoBox>
    </DocPage>
  );
}
