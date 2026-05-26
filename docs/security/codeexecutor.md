# Secure Code Execution — CodeExecutor & Sandbox

## Threat Model

When an agent receives an attacker-controlled prompt, it may produce arbitrary
Python/shell code and pass it to `CodeExecutor`. Without isolation, this code
runs directly on the host with the privileges of the effGen process.

**Attack surface:** Prompt injection → agent emits `os.system("rm -rf /")` or
exfiltrates secrets via network calls.

**Mitigations provided by the sandbox layer:**

| Control | DockerSandbox | SubprocessSandbox |
|---|---|---|
| Network isolation | ✅ `--network=none` | ✅ `unshare --map-root-user --net` (Linux, no privileges) |
| Filesystem isolation | ✅ `--read-only` | ⚠️ private tmpfs over `/tmp` via `unshare --mount` (rest of host FS readable) |
| Memory cap | ✅ `--memory=256m` | ⚠️ `ulimit -v` (advisory) |
| CPU cap | ✅ `--cpus=1` | ⚠️ `ulimit -t` (advisory) |
| PID limit | ✅ `--pids-limit=100` | ⚠️ `ulimit -u 256` |
| Capability drop | ✅ `--cap-drop=ALL` | ⚠️ unprivileged user namespace (no host root) |
| Privilege escalation | ✅ `--no-new-privileges` | ⚠️ user-namespace UID mapping |

> The SubprocessSandbox network and `/tmp` isolation use **unprivileged user
> namespaces** — they do **not** require `CAP_SYS_ADMIN` or root. If user
> namespaces are disabled on the host, the sandbox degrades to `ulimit`-only
> mode and logs a warning naming exactly which protections are inactive.

---

## Sandbox Backends

### DockerSandbox (Default when Docker available)

Runs code in a container built from `effgen/sandbox:python-3.11` (or the
stock `python:3.11-slim` if the custom image is not present locally).

```text
docker run --rm
  --read-only
  --tmpfs /tmp:size=64m,noexec
  -v <tmpdir>:/workspace:ro
  -w /workspace
  --network none
  --cap-drop ALL
  --no-new-privileges
  --security-opt no-new-privileges:true
  --pids-limit 100
  --memory 256m
  --memory-swap 256m
  --cpus 1
  effgen/sandbox:python-3.11
  python3 script.py
```

Build the custom sandbox image:

```bash
docker build -f deploy/sandbox/Dockerfile.sandbox \
             -t effgen/sandbox:python-3.11 .
```

### SubprocessSandbox (Fallback)

Used when Docker daemon is unreachable. Provides **partial isolation** via an
unprivileged user namespace. Code is fed to the interpreter over **stdin**, so
no script file is written to the host filesystem.

On Linux with unprivileged user namespaces available:

```bash
unshare --map-root-user --mount --net bash -c \
  "ulimit -v 262144 -t 10 -u 256; mount -t tmpfs none /tmp; exec python3 -"
```

- `--map-root-user` — root inside the namespace only; no host privileges.
- `--net`           — fresh network namespace with no interfaces → outbound
                      network blocked (DNS + connect fail). No `CAP_SYS_ADMIN`.
- `--mount` + tmpfs — a private tmpfs over `/tmp`, so `rm -rf /tmp/...` and other
                      writes/deletes under `/tmp` never touch the host.

**Caveats:**
- Filesystem isolation only shields `/tmp`; the rest of the host FS remains
  *readable* (though not writable as host-root). Use DockerSandbox for full FS
  isolation.
- Requires unprivileged user namespaces (`kernel.unprivileged_userns_clone=1`
  / `user.max_user_namespaces>0`). When unavailable, the sandbox degrades to
  `ulimit`-only mode and logs a warning listing the inactive protections.
- Memory limit is advisory (`ulimit -v`), not enforced by cgroups.
- A loud `WARNING` is emitted at startup when this fallback is selected as the
  default.

### OffSandbox (Explicit opt-out — UNSAFE)

Set `EFFGEN_SANDBOX_BACKEND=off` to disable sandboxing entirely. Code then runs
directly on the host with the privileges of the effGen process. A loud warning
is emitted on every execution, and this backend is **never** chosen by
auto-resolution. Use only for fully trusted code in controlled environments.

### FirecrackerSandbox (Stub — not implemented)

Interface stub for future Firecracker MicroVM integration. Raises
`NotImplementedError` if called. Full implementation requires:
- Firecracker binary + KVM device
- A minimal guest rootfs (Alpine initrd)
- `jailer` for additional isolation
- vsock/REST API for code injection

---

## Configuration

All options are environment-driven:

| Environment Variable | Default | Description |
|---|---|---|
| `EFFGEN_SANDBOX_BACKEND` | `auto` | `auto`, `docker`, `subprocess`, `firecracker`, `off` (unsafe) |
| `EFFGEN_SANDBOX_TIMEOUT` | `10` | Execution timeout in seconds |
| `EFFGEN_SANDBOX_MEMORY` | `256m` | Memory limit (Docker-style: `256m`, `1g`) |
| `EFFGEN_SANDBOX_IMAGE` | `effgen/sandbox:python-3.11` | Custom Docker image to use |

**Auto-selection:** When `EFFGEN_SANDBOX_BACKEND=auto` (default), effGen picks
DockerSandbox if the Docker daemon is reachable; otherwise SubprocessSandbox
with a warning.

Per-call overrides (passed to `CodeExecutor`):

```python
result = await executor._execute(
    code="print('hi')",
    language="python",
    timeout=30,
    memory_limit="512m",
    network_enabled=False,
)
```

---

## Quick Start

### Using CodeExecutor in an Agent

```python
from effgen import Agent, AgentConfig
from effgen.tools.builtin.code_executor import CodeExecutor

agent = Agent(
    config=AgentConfig(provider="openai", model="gpt-4o-mini"),
    tools=[CodeExecutor()],
)

response = await agent.run(
    "Write and run a Python script that computes the first 10 prime numbers."
)
print(response.content)
```

### Direct Sandbox Usage

```python
import asyncio
from effgen.security.sandbox import get_sandbox, SandboxConfig

async def main():
    config = SandboxConfig(
        backend="auto",   # docker if available, else subprocess
        timeout=15,
        memory_limit="256m",
        network_enabled=False,
    )
    sandbox = await get_sandbox(config)
    result = await sandbox.run(
        code="print(sum(range(100)))",
        language="python",
        config=config,
    )
    print(f"stdout: {result.stdout}")
    print(f"backend: {result.backend_used}")

asyncio.run(main())
```

---

## Building the Sandbox Image

```bash
# Build
docker build \
  -f deploy/sandbox/Dockerfile.sandbox \
  -t effgen/sandbox:python-3.11 \
  .

# Smoke-test network isolation
docker run --rm --network=none effgen/sandbox:python-3.11 \
  python3 -c "
import urllib.request, sys
try:
    urllib.request.urlopen('https://example.com', timeout=3)
    print('FAIL: network was accessible')
    sys.exit(1)
except Exception as e:
    print(f'PASS: network blocked ({type(e).__name__})')
    sys.exit(0)
"
```

---

## Security Recommendations

1. **Always use DockerSandbox in production.** Install Docker and add the
   effGen service account to the `docker` group.
2. **Do not set `network_enabled=True`** unless the task explicitly requires
   network access and you have reviewed the code.
3. **Pin the sandbox image** to a specific digest in production environments:
   ```bash
   EFFGEN_SANDBOX_IMAGE=effgen/sandbox:python-3.11@sha256:<digest>
   ```
4. **Monitor container resource usage** via Docker stats or cAdvisor to detect
   abuse.
5. **Treat SubprocessSandbox as a development-only fallback.** Never rely on it
   for production workloads handling untrusted code.

---

## Limitations & Future Work

- **Firecracker integration** (`FirecrackerSandbox`) — planned for a future
  release; provides VM-level isolation with sub-second boot times.
- **Language support** — currently Python, JavaScript (Node), Bash/sh.
  Adding Rust, Go, Ruby planned.
- **seccomp profiles** — custom `--seccomp-profile` for DockerSandbox to
  further restrict syscalls (e.g., block `ptrace`, `mount`).
- **gVisor** (`runsc`) runtime — alternative to Firecracker for stronger
  syscall interposition without full VM overhead.
