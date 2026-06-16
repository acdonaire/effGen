# Contributing to effGen

Thank you for your interest in contributing to effGen! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (for integration tests)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/ctrl-gaurav/effGen.git
cd effGen

# Create a virtual environment
conda create -n effgen python=3.11 -y
conda activate effgen

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Unit tests (no GPU required)
pytest tests/unit/ -v --no-cov

# Integration tests (requires GPU)
CUDA_VISIBLE_DEVICES=0 pytest tests/integration/ -v -m gpu --no-cov

# Performance benchmarks
pytest tests/benchmarks/ -v --no-cov

# All tests with coverage
pytest tests/ -v
```

### Cleaning up artifacts

Test, build, and runtime runs leave gitignored artifacts behind (caches, coverage,
`./checkpoints`, `__pycache__`, etc.). Tidy the tree at any time with:

```bash
scripts/clean.sh            # remove caches, coverage, build artifacts, runtime state
scripts/clean.sh --dry-run  # preview what would be removed
```

It only deletes already-ignored artifacts — never tracked source, `.git/`, or `.env`.

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting (profile: black)
- **flake8** for linting
- **mypy** for type checking
- **bandit** for security linting

All these run automatically via pre-commit hooks. You can also run them manually:

```bash
black effgen/
isort effgen/
flake8 effgen/
mypy effgen/ --ignore-missing-imports
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for your changes (unit tests at minimum, integration tests if GPU-dependent)
3. **Update CHANGELOG.md** with a summary of your changes
4. **Ensure all tests pass**: `pytest tests/unit/ -v --no-cov`
5. **Submit your PR** with a clear description of the changes

### PR Checklist

- [ ] Tests added/updated
- [ ] CHANGELOG.md updated
- [ ] Code passes `black --check` and `isort --check`
- [ ] No new `TODO`/`FIXME` without a tracking issue
- [ ] Documentation updated if public API changed

## Issue Reporting

When reporting bugs, please include:

- Python version (`python --version`)
- effGen version (`python -c "import effgen; print(effgen.__version__)"`)
- GPU info (if relevant): `nvidia-smi`
- Full error traceback
- Steps to reproduce

## Architecture Overview

```
effgen/
├── core/           # Agent, AgentConfig, ReAct loop
├── models/         # Model backends (vLLM, Transformers, API adapters)
├── tools/          # Built-in tools and protocols (MCP, A2A, ACP)
├── memory/         # Short-term, long-term, vector memory
├── prompts/        # Template management and optimization
├── config/         # Configuration loading and validation
├── execution/      # Code execution sandboxing
├── gpu/            # GPU allocation and monitoring
└── utils/          # Logging, metrics, validators, health checks
```

### Key Design Principles

- **Open Source First**: All features must work without paid APIs
- **SLM-Optimized**: Prompts and tools designed for 1B-7B parameter models
- **Tools extend `BaseTool`** with `async _execute()` method
- **Agent uses ReAct loop**: Thought → Action → Observation → repeat

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
