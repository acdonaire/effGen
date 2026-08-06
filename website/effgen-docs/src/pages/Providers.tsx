import React from 'react';
import { Server } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Providers() {
  return (
    <DocPage
      title="Providers"
      subtitle="v0.2.1 provider loading for OpenAI, Anthropic, Gemini, and Cerebras, plus v0.2.3+ ProviderRegistry, auth checks, and backend parity. v0.3.0 adds a self-updating, drift-aware model catalog and a fail-closed error taxonomy across every provider."
      icon={<Server size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Providers' },
      ]}
    >
      <InfoBox type="success" title="v0.3.0 — self-updating catalog &amp; typed provider errors">
        <p>
          Every provider now ships a priced, dated catalog snapshot. <code>effgen models refresh</code>{' '}
          diffs the live API and reports models added / removed / changed, and <code>check_drift()</code>{' '}
          warns once when the snapshot looks stale — see <a href="/docs/models">Models</a>. Provider
          failures are also typed and fail closed: <code>classify_provider_error()</code> maps
          exceptions to <code>metadata["error"]["category"]</code> — <code>auth</code> /{' '}
          <code>not_found</code> / <code>rate_limited</code> / <code>transient</code> /{' '}
          <code>timeout</code> / <code>fatal</code> — retries fire only when they can help, and a
          wrong / 404 model id suggests the nearest live alternative instead of crashing. Use{' '}
          <code>effgen doctor --live --cheap</code> to see which keys are actually usable, not just
          present.
        </p>
      </InfoBox>

      <h2>v0.2.1 Provider Loading</h2>
      <p>
        In v0.2.1, <code>provider=</code> is supported for cloud routing.
        OpenAI, Anthropic, and Gemini are also auto-detected from their model
        names; Cerebras should use explicit <code>provider="cerebras"</code>.
        Provider-prefixed IDs and <code>effgen doctor</code> arrive in v0.2.3+.
      </p>
      <CodeBlock
        code={`from effgen import load_model

openai = load_model("gpt-5.4-nano", provider="openai")
anthropic = load_model("claude-3-5-sonnet-20241022", provider="anthropic")
gemini = load_model("gemini-2.0-flash-lite", provider="gemini")
cerebras = load_model("llama3.1-8b", provider="cerebras")`}
        language="python"
        filename="v021_providers.py"
      />

      <h2>ProviderRegistry (v0.2.3+)</h2>
      <p>
        The <code>ProviderRegistry</code> is the singleton source of truth for cloud
        adapters and model metadata. Adapters self-register when imported, and
        <code> load_model</code> uses the registry for <code>provider:model_id</code>
        prefix parsing.
      </p>
      <CodeBlock
        code={`from effgen.models.registry import list_providers, list_models, lookup
from effgen.models.auth import check_keys

providers = list_providers()
models = list_models("groq")
provider, adapter_cls, info = lookup("groq:llama-3.1-8b-instant")
keys = check_keys()`}
        language="python"
        filename="provider_registry.py"
      />

      <FeatureList
        features={[
          { icon: 'R', title: 'Registration', description: 'All 9 cloud adapters register idempotently on import.' },
          { icon: 'L', title: 'Lookup', description: 'Use provider:model_id prefixes to avoid AmbiguousModelError for shared model IDs.' },
          { icon: 'A', title: 'Auth', description: 'check_keys() and effgen doctor report which provider keys are ready in v0.2.3+.' },
          { icon: 'E', title: 'Errors', description: 'ModelAuthError is uniform across all 9 cloud providers; typed not-found, timeout, and unavailable errors are surfaced where applicable.' },
        ]}
      />

      <h2>Provider Matrix by Version</h2>
      <ApiTable
        headers={['Provider', 'Extra', 'Env key', 'Models', 'Streaming', 'Native tools', 'Rate / cost']}
        rows={[
          [<code>openai</code>, 'built in', <code>OPENAI_API_KEY</code>, '35', 'Yes', 'Yes', 'Provider pricing + usage metadata'],
          [<code>anthropic</code>, 'built in', <code>ANTHROPIC_API_KEY</code>, 'v0.2.1 adapter; v0.2.2+: 17 Claude entries', 'Yes', 'v0.2.2+ experimental adapter specs', 'Provider pricing; v0.2.2+ cache usage'],
          [<code>gemini</code>, 'built in', <code>GOOGLE_API_KEY</code>, 'v0.2.1 adapter; v0.2.2+: 16 Gemini/Gemma entries', 'Yes', 'v0.2.2+ native tools', 'Provider quotas; v0.2.2+ grounding usage'],
          [<code>cerebras</code>, <code>effgen[cerebras]</code>, <code>CEREBRAS_API_KEY</code>, '4 registered; 2 reliably free-tier callable', 'Yes', 'Model-dependent', 'RateLimitCoordinator + CostTracker ($0 for free-tier calls)'],
          [<code>groq</code>, <code>effgen[groq]</code>, <code>GROQ_API_KEY</code>, 'v0.2.3+: 16', 'Yes', 'Yes', 'RPM/RPD/TPM/TPD windows + CostTracker ($0 free tier)'],
          [<code>together</code>, <code>effgen[together]</code>, <code>TOGETHER_API_KEY</code>, 'v0.2.3+: 149 chat models registered; 163 total catalog incl. 13 language + 1 embedding', 'Yes', 'Yes', 'Rate-limit coordination + per-model pricing'],
          [<code>fireworks</code>, <code>effgen[fireworks]</code>, <code>FIREWORKS_API_KEY</code>, 'v0.2.3+: 80', 'Yes', '54 tool-capable', 'Rate-limit coordination + per-model pricing'],
          [<code>replicate</code>, <code>effgen[replicate]</code>, <code>REPLICATE_API_TOKEN</code>, 'v0.2.3+: 38', '34 streaming-capable', 'Limited', 'Timeout cancellation + compute_seconds metadata'],
          [<code>hf</code>, <code>effgen[hf]</code>, <code>HF_TOKEN</code>, 'v0.2.3+: 124', 'Yes', 'Yes for tool-capable models', 'Router availability + helpful fallback errors'],
        ]}
      />

      <h2>Load by Provider</h2>
      <CodeBlock
        code={`from effgen import load_model

openai = load_model("gpt-5.4-nano", provider="openai")  # v0.2.1-compatible
gemini = load_model("gemini-2.0-flash-lite", provider="gemini")  # v0.2.1-compatible
cerebras = load_model("llama3.1-8b", provider="cerebras")
gemini_25 = load_model("gemini-2.5-flash", provider="gemini")  # v0.2.2+
groq = load_model("groq:llama-3.3-70b-versatile")  # v0.2.3+ prefix
# v0.2.3+
together = load_model("meta-llama/Meta-Llama-3-8B-Instruct-Lite", provider="together")
fireworks = load_model("accounts/fireworks/models/qwen3-8b", provider="fireworks")
replicate = load_model("meta/meta-llama-3-8b-instruct", provider="replicate")
hf = load_model("Qwen/Qwen2.5-72B-Instruct", provider="hf")`}
        language="python"
        filename="load_by_provider.py"
      />

      <h2>effgen doctor (v0.2.3+)</h2>
      <p>
        <code>effgen doctor</code> loads <code>~/.effgen/.env</code> and the project
        <code> .env</code>, then reports whether each provider has an API key available.
      </p>
      <CodeBlock
        code={`effgen doctor
effgen doctor --provider hf
effgen doctor --json`}
        language="bash"
        filename="terminal"
      />

      <h2>Backend Parity</h2>
      <p>
        The v0.2.3 parity suite validates the canonical agentic calculator task
        <code> (17 * 23) + sqrt(144) = 403</code> across available providers.
      </p>
      <ApiTable
        headers={['Capability', 'Validated result']}
        rows={[
          ['ReAct task execution', '7/8 providers returned the expected answer: 403; Anthropic skipped because no key was available; Replicate marked billing-credit xfail'],
          ['Streaming', '7/7 validated streaming providers yielded token chunks'],
          ['Auth errors', '9/9 cloud providers raise ModelAuthError for invalid credentials'],
          ['Native tool strategy', 'Provider/model-supported native tool paths are tracked in the matrix; not every model exposes native tools'],
        ]}
      />

      <h2>Cerebras Registered Models</h2>
      <ApiTable
        headers={['Model', 'Free tier', 'Native tools', 'Context / output', 'RPM/RPH/RPD', 'TPM/TPH/TPD', 'Limits / cost']}
        rows={[
          [<code>gpt-oss-120b</code>, 'Restricted', 'Yes', '65,536 / 32,768', '30/900/14,400', '64k/1M/1M', 'Tracked by RateLimitCoordinator and CostTracker; commonly access-restricted on free tier'],
          [<code>llama3.1-8b</code>, 'Yes', 'Yes', '8,192 / 8,192', '30/900/14,400', '60k/1M/1M', 'Free-tier pricing recorded as $0; scheduled deprecation May 27, 2026'],
          [<code>qwen-3-235b-a22b-instruct-2507</code>, 'Yes', 'Yes', '65,536 / 32,768', '30/900/14,400', '60k/1M/1M', 'Daily budget exhaustion raises RateLimitExceeded; scheduled deprecation May 27, 2026'],
          [<code>zai-glm-4.7</code>, 'Restricted', 'No', '65,536 / 40,960', '10/100/100', '60k/1M/1M', 'Streaming preserves usage on the terminal chunk when available'],
        ]}
      />

      <CodeBlock
        code={`from effgen.models.cerebras_models import available_models, deprecated_models, free_tier_models, model_info
from effgen.models._cost import CostTracker
from effgen.models._rate_limit import RateLimitCoordinator, RateLimitExceeded

print(available_models())
print(free_tier_models())
print(deprecated_models())
print(model_info("llama3.1-8b"))
print(CostTracker.get().summary())`}
        language="python"
        filename="cerebras_registry.py"
      />

      <InfoBox type="info" title="Ambiguous model IDs (v0.2.3+)">
        <p>
          Model IDs such as <code>llama-3.3-70b-versatile</code> can appear on multiple
          providers. Use <code>groq:llama-3.3-70b-versatile</code> or
          <code> provider="groq"</code> to make routing explicit. Bare ambiguous IDs raise
          <code> AmbiguousModelError</code> with the candidate provider list.
        </p>
      </InfoBox>

      <h2>Provider Capability Matrix (v0.2.4+)</h2>
      <p>
        v0.2.4 surfaces a provider-level capability matrix through
        <code> ProviderRegistry.get_capabilities(provider)</code>. The
        <code> PolicyBasedRouter</code> consults this matrix plus per-model registry
        flags (<code>supports_streaming</code>, <code>supports_native_tools</code>,
        <code> supports_vision</code>, <code>supports_grounding</code>,
        <code> supports_thinking</code>, <code>modality</code>,
        <code> requires_endpoint</code>) to eliminate ineligible candidates and
        record the reason on every <code>RouterDecision</code>.
      </p>
      <ApiTable
        headers={['Provider', 'chat', 'tools', 'streaming', 'vision', 'grounding', 'thinking', 'json_schema']}
        rows={[
          ['cerebras', '✓', '✓', '✓', '', '', '', '✓'],
          ['openai',   '✓', '✓', '✓', '✓', '', '✓', '✓'],
          ['groq',     '✓', '✓', '✓', '', '', '', '✓'],
          ['anthropic','✓', '✓', '✓', '✓', '', '✓', ''],
          ['gemini',   '✓', '✓', '✓', '✓', '✓', '✓', '✓'],
          ['together', '✓', '✓', '✓', '', '', '', '✓'],
          ['fireworks','✓', '✓', '✓', '', '', '', '✓'],
          ['replicate','✓', '',  '✓', '✓', '', '', ''],
          ['hf',       '✓', '',  '✓', '', '', '', ''],
        ]}
      />

      <h2>Policy-Based Router (v0.2.4+)</h2>
      <p>
        The <code>PolicyBasedRouter</code> picks the best
        <code> (provider, model_id)</code> pair for a request. Pass
        <code> RoutingContext(prompt_tokens_estimate, user_budget_usd,
        latency_budget_ms, required_capabilities)</code> and chain
        <code> FirstAvailablePolicy</code>, <code>CostBasedPolicy</code>, or
        <code> LatencyBasedPolicy</code>. <code>route_and_execute(context, fn)</code>
        transparently retries on rate-limits, 5xx errors, timeouts, and
        <code> BudgetExceededError</code>, emitting one <code>RouterEvent</code> per
        failover hop. See the Models doc for the full recipe.
      </p>
      <CodeBlock
        code={`import effgen.models  # register all 9 adapters
from effgen import PolicyBasedRouter, RoutingContext, CostBasedPolicy
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(policies=[CostBasedPolicy()])
decision = router.route(RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=0.001,
    required_capabilities={Capability.chat},
))
print(decision.chosen.provider, decision.chosen.model_id)
print(decision.policy_name, decision.score)
for pair, reason in decision.eliminated:
    print(" -", pair.provider, pair.model_id, "->", reason)`}
        language="python"
        filename="router_decision.py"
      />
    </DocPage>
  );
}
