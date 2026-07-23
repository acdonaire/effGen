# Terminal appearance

This page documents how the `effgen` command-line tool decides what to draw:
color, the logo and first-run welcome, live animation, and the branded landing
shown when you run `effgen` with no command. Every visual is TTY-only and opt-out;
piped, redirected, and `--json` output stays plain text and is unaffected by the
settings below.

## Color and themes

`effgen` ships four color themes. The default look is unchanged from earlier
releases; the others are opt-in.

| Theme | Use it for |
|---|---|
| `default` | the standard look |
| `high-contrast` | bright, bold hues and no dim text (low-vision readers) |
| `monochrome` | structure only — bold/underline carry meaning, no hue |
| `light` | tuned for a light terminal background |

Select a theme for one command with `--theme`, or for the session with the
`EFFGEN_THEME` environment variable:

```bash
effgen --theme high-contrast doctor
export EFFGEN_THEME=light
```

`--theme` may appear before or after the subcommand.

### Turning color off

`effgen` honors the [`NO_COLOR`](https://no-color.org/) convention. Set
`NO_COLOR` to any value and every command renders structure — tables, panels,
the logo — with no color, whatever theme is selected:

```bash
NO_COLOR=1 effgen models list
```

Color is also off automatically when output is not a terminal (a pipe, a file, a
CI log) and when `TERM=dumb`.

## The bare `effgen` landing

Running `effgen` with no command **on an interactive terminal** shows a short
landing: the logo, the version, and a handful of quick actions. Press **Enter**
to continue into the interactive setup wizard (the historical behavior), or pick
a quick action:

```
  [Enter]  set up and run an agent (interactive wizard)
  [c]      chat — open an interactive session
  [q]      quickstart — a 2-minute guided first run
  [d]      doctor — check which providers are ready
  [m]      models — browse the model catalog
  [h]      help — all commands            [x] exit
```

The landing appears only when both input and output are terminals. When output
is piped or redirected, when you pass `--quiet`, or in CI, `effgen` skips the
landing and behaves exactly as before. The logo uses a Unicode block banner on
terminals that can render it and a plain ASCII banner otherwise; it collapses to
a one-line wordmark on a narrow terminal.

## The first-run welcome

The first time you use `effgen` on an interactive terminal, it prints a one-time
welcome pointing at `effgen doctor` and a hello agent, then records a flag under
`~/.effgen` so it never shows again. It is silent under `--quiet`, in CI, and on
non-interactive output.

## Live animation

Long-running commands show a live status line and progress bars on an
interactive terminal. Turn the animation off — leaving plain, single-line status
text that is safe to pipe or log — in any of these ways:

```bash
effgen run "..." --no-animation
export EFFGEN_NO_ANIM=1
```

`NO_COLOR` and a non-interactive stdout also disable animation.

## Tips

`effgen` surfaces an occasional one-line tip after commands you watch
interactively. Silence them with:

```bash
export EFFGEN_TIPS=0
```

## Summary of environment variables

| Variable | Effect |
|---|---|
| `NO_COLOR` | disable all color (structure still renders) |
| `EFFGEN_THEME` | select a theme (`default`, `high-contrast`, `monochrome`, `light`) |
| `EFFGEN_NO_ANIM` | disable live spinners/progress animation |
| `EFFGEN_TIPS=0` | silence the rotating tips |
| `EFFGEN_HOME` | relocate the per-user state directory (default `~/.effgen`) |
