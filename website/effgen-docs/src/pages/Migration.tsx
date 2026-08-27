import { ArrowUpCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Steps,
  Step,
  Terminal,
} from '../components/docs';
import { pythonVersions, publicNameCount, version } from '../siteData';

export default function Migration() {
  return (
    <DocPage
      subtitle={`The three breaking changes in ${version} and what each one asks you to change.`}
      icon={<ArrowUpCircle size={48} />}
    >
      <p>
        {version} is the first stable release. The public surface grew to {publicNameCount} names
        and nothing was removed or renamed, so most code upgrades by changing the version. Three
        changes are breaking, and each has a one-line migration.
      </p>

      <h2>The upgrade</h2>

      <CodeBlock
        language="bash"
        code={`pip install --upgrade effgen
python -c "import effgen; print(effgen.__version__)"`}
      />

      <Terminal
        command={'python -c "import effgen; print(effgen.__version__)"'}
        output={version}
      />

      <h2>1. Python 3.10 is no longer supported</h2>
      <p>
        The floor is 3.11, and the supported set is {pythonVersions.join(', ')}.{' '}
        <code>tomllib</code>, <code>asyncio.timeout</code>, <code>datetime.UTC</code> and the{' '}
        <code>TimeoutError</code> unification are all standard library from 3.11, and effGen
        carried a hand-written fallback for each.
      </p>
      <p>
        <strong>Migration:</strong> upgrade the interpreter. Nothing in the API changed.
      </p>

      <Callout type="note" title="On 3.14, the [all] extra needs a lock file">
        <p>
          The base install and every named extra install normally on 3.14. Only{' '}
          <code>effgen[all]</code> needs{' '}
          <code>pip install -r requirements-all-py314-lock.txt</code> followed by{' '}
          <code>pip install --no-deps effgen</code> —{' '}
          <Link to="/installation">Installation</Link> explains why.
        </p>
      </Callout>

      <h2>2. raise_on_error now defaults to True</h2>
      <p>
        A failed run raises its typed error instead of returning an{' '}
        <code>AgentResponse</code> with <code>success=False</code> and a plausible-looking string
        in <code>.output</code> — which a caller reading <code>.output</code> without checking{' '}
        <code>.success</code> never noticed.
      </p>
      <p>
        <strong>Migration:</strong> pass <code>raise_on_error=False</code> to inspect the response
        yourself. The failure shape is unchanged.
      </p>

      <CodeBlock code={`from effgen import Agent, AgentConfig

Agent(AgentConfig(model="openai:gpt-5-nano", raise_on_error=False))`} />

      <Callout type="warning" title="With the flag off, read partial_output rather than output">
        <p>
          A failed run's <code>output</code> is effGen's report of what stopped it. The model's own
          text is in <code>metadata["partial_output"]</code>.
        </p>
      </Callout>

      <CodeBlock
        filename="inspect.py"
        code={`from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[Calculator()],
    tool_calling_mode="react",
    max_iterations=1,
    temperature=0.0,
    raise_on_error=False,          # the 1.0.0 default is True
))
r = agent.run("What is 4817 * 236? Use the calculator, then explain the result.")

print("success:", r.success)
print("reason:", r.metadata.get("reason"))
print("output:", r.output[:90])
print("the model's own text:", r.metadata.get("partial_output"))`}
      />

      <Terminal command="python inspect.py" output={`success: False
reason: max_iterations_partial
output: Stopped after 1 iteration without a final answer: 'gpt-5-nano' was still taking tool steps
the model's own text: 1136812`} />

      <Callout type="tip" title="Batch evaluation wants raise_on_error=False">
        <p>
          Scoring a run that hit the iteration cap as an error rather than as a wrong answer
          measures the reporting style instead of the model, and a small model hits that cap often.
          What makes turning the flag off safe is the change below: an ordinary failure comes back
          to be inspected, while a backend that was never reached still raises — so a broken
          endpoint cannot be silently scored as a wrong answer. The command line does exactly this
          at all fourteen of its construction sites.
        </p>
      </Callout>

      <h2>3. A backend that never answered raises</h2>
      <p>
        A refused connection, an unresolvable host or a missing route is classified{' '}
        <strong>unreachable</strong> — separately from a server that answered badly, which stays
        transient and is still retried — and raises <code>BackendUnreachableError</code>.
      </p>
      <p>
        <strong>Migration:</strong> there is no opt-out, by design. Catch the error where you want
        to handle it.
      </p>

      <CodeBlock
        filename="unreachable.py"
        code={`from effgen import Agent, AgentConfig
from effgen.models.errors import BackendUnreachableError

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:9/v1",     # nothing is listening here
))

try:
    agent.run("What is 6 times 7?")
except BackendUnreachableError as e:
    print(type(e).__name__)
    print(str(e)[:220])`}
      />

      <Terminal command="python unreachable.py" output={`BackendUnreachableError
openai did not answer (model='Qwen/Qwen2.5-7B-Instruct'): OpenAI generation failed [will_retry]: Connection error.. Nothing answered at that endpoint — check the server is running and the base_url, host and port are righ`} />

      <p>
        A task that ran and failed is a result you can inspect. A backend that was never reached is
        not, and returning one is how a whole batch completes against nothing and still looks
        healthy in the summary. Classification reads the exception chain, because provider SDKs
        shorten a refused port to "Connection error." and keep the real cause on{' '}
        <code>__cause__</code>.
      </p>

      <h2>One smaller change worth knowing</h2>
      <p>
        Four enums are now <code>enum.StrEnum</code>: <code>TaskStatus</code> (from{' '}
        <code>effgen.core.background</code>), <code>AlertSeverity</code>,{' '}
        <code>PermissionMode</code> and <code>LoadScenario</code>. Equality, membership and JSON
        serialization are unchanged; what changed is that <code>str()</code> now gives the value
        rather than <code>ClassName.MEMBER</code>.
      </p>

      <CodeBlock
        filename="strenum.py"
        code={`from effgen.core.background import TaskStatus

print(str(TaskStatus.RUNNING))            # "RUNNING" since 1.0.0
print(TaskStatus.RUNNING == "RUNNING", TaskStatus.RUNNING in ("RUNNING", "FAILED"))`}
      />

      <Terminal
        command="python strenum.py"
        output={`RUNNING
True True`}
        caption="Anything that formatted one of these into a log line or a filename gets a shorter string than it used to."
      />

      <Callout type="note" title="Two different TaskStatus classes exist">
        <p>
          The one exported from the top-level package is a plain <code>Enum</code> used by the
          multi-agent task graph. The <code>StrEnum</code> is{' '}
          <code>effgen.core.background.TaskStatus</code>, which is what background jobs report. The
          import path decides which you get.
        </p>
      </Callout>

      <h2>Upgrading, in order</h2>

      <Steps>
        <Step title="Move to Python 3.11 or newer">
          <p>Nothing in the API changed with it, so do this first and separately.</p>
        </Step>
        <Step title="Find every place that reads .output without checking .success">
          <p>
            Those are the call sites the second change affects. Either let them raise — which is
            usually what you want — or pass <code>raise_on_error=False</code> and check{' '}
            <code>.success</code> explicitly.
          </p>
        </Step>
        <Step title="Decide where an unreachable backend should be handled">
          <p>
            A batch runner, a job queue and a server all want to handle it differently. Catching{' '}
            <code>BackendUnreachableError</code> at the boundary is usually right.
          </p>
        </Step>
        <Step title="Search for str() on the four enums">
          <p>Only if you format them into logs, filenames or a wire format.</p>
        </Step>
      </Steps>

      <h2>What did not change</h2>

      <ApiTable
        headers={['Concern', 'Status']}
        rows={[
          ['Public names', `Grown to ${publicNameCount}. Nothing removed, nothing renamed.`],
          [
            <code>Agent</code>,
            <>
              <code>AgentConfig</code>, <code>load_model</code> and every tool API work unchanged.
            </>,
          ],
          [
            <code>AgentResponse.tool_calls</code>,
            'Now the calls themselves rather than only a count — but it still compares and casts as the count, so tool_calls == 2 and tool_calls > 0 are unchanged, and to_dict() keeps the count under its original key.',
          ],
          ['The failure shape', 'Unchanged. It is when you get it that changed.'],
          ['Configuration files', 'Unchanged.'],
          ['The server API', 'Unchanged for existing endpoints.'],
        ]}
      />

      <h2>Coming from another framework</h2>
      <p>
        If you are not upgrading but arriving — from the OpenAI SDK, from LangChain, or from
        anything that speaks the OpenAI protocol — the route in is effGen's own server, which most
        client code reaches by changing only its <code>base_url</code>.{' '}
        <Link to="/openai-api">OpenAI-compatible API</Link> is the full endpoint, alias, streaming
        and error-status reference, and <Link to="/clients">Clients and SDKs</Link> covers the
        lighter native client. The one place the protocol differs is tools: effGen runs its own
        registered tools on the server and returns the final answer, rather than forwarding
        client-defined function tools back for the caller to run.
      </p>

      <SeeAlso paths={['/releases', '/errors', '/installation']} />
    </DocPage>
  );
}
