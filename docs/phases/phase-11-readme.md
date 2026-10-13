# Phase 11: A creative interface, a dashboard, and reproducibility

## What we built
| Piece | Where |
|---|---|
| Dark theme, gradient hero with animated shield, glass cards, animated score rings, result cards, chips, threshold ticks | `.streamlit/config.toml`, `src/ui/theme.py`, `src/ui/components.py` |
| Example buttons on every detector | `app.py` (`example_row`) |
| Dashboard tab: KPIs, generalisation charts, per-label chart, data sizes, speed/size, weaknesses, session tally, training charts | `src/ui/dashboard.py` |
| One-command pipeline plan with input checks | `scripts/rebuild_all.py` |
| Speed and size benchmark | `scripts/benchmark_models.py` -> `reports/metrics/performance.csv` |
| 33 new tests (components, dashboard, rebuild plan, security) | `tests/test_ui_components.py`, `tests/test_dashboard.py`, `tests/test_rebuild_plan.py`, `tests/test_security.py` |

Screenshots of the result are in `docs/images/` and in the README.

## Design decisions and why
- **Same safety rules as before.** We still never use `unsafe_allow_html`. All markup goes through `st.html` (sanitised); text that is not ours is escaped; a test parses every block for script tags and event attributes.
- **No third-party requests.** No web fonts or CDN images. Art is drawn in SVG by us.
- **The dashboard teaches, not decorates.** Its centrepiece compares the same-source test with a different-source test. A model that is 99% on its own test and 69% on real 2026 spam is the most important thing a reader can learn about this project.
- **Honest numbers only.** Charts read `metadata.json`; weaknesses are shown as numbers; the session tally holds counts only.
- **Animations are optional.** `prefers-reduced-motion` turns them off; the native alerts remain beside the custom result card.

## Difficulties (what went wrong and how it was found)
1. **Blank illustrations.** My first version used inline `<svg>`. All tests passed, yet the page had empty boxes. Streamlit's sanitiser removes inline SVG. Only a real browser screenshot (Playwright) showed it. Fix: SVG documents embedded as `<img>` data URIs, which also keep their own animations.
2. **Black-on-dark text.** The custom CSS assumed a dark page, but the app theme was still light. Setting `[theme]` in `config.toml` fixed it.
3. **Every click took about 2.5 seconds.** Streamlit runs all tabs on every rerun, so the dashboard's charts were redrawn on each click. My own browser test clicked "Analyze" before the example text arrived and looked like a bug. The Dashboard is now drawn only when its tab is open.
4. **Tests that used `button[0]`.** Example buttons come first on the page, so those tests clicked the wrong button. They now use widget keys.
5. **A test that failed for the wrong reason.** A crude regex flagged the word `onerror` inside an escaped attribute. The test now parses real tags and attributes.
6. **Benchmark noise.** The first benchmark ran while other jobs were busy (URL model load 13 s). Always benchmark on an idle machine and say where it was measured.

## Results
- 389 tests pass (3 expected failures and 1 unexpected pass document known weaknesses); coverage of the serving code and UI is 95%; the linter is clean.
- Speed and size are in `reports/metrics/performance.csv` and on the Dashboard. Typical single checks take a few milliseconds; the slowest 5% for emails and comments are far slower because of the explanation step on long inputs.

## What this phase does not do
- No light theme and no language switch.
- Animations were checked in Chromium only.
- A browser screenshot test is manual (Playwright is not a project dependency).

## Self-check
1. Why can all tests pass while the page is visually broken? What did we use to find out?
2. Why is an SVG shown through `<img>` safer than inline SVG?
3. What does the gap between the two bars in the dashboard's first chart mean?
4. Why render the dashboard only when its tab is open?

## Done when
The app looks and feels like a product, shows its own evidence and weaknesses, makes no third-party requests, and every new piece is tested and documented.

## Update: theme change to Crimson Noir
You picked Crimson Noir from six previews. What it took: colours moved from fixed values into one palette; text boxes, charts, illustrations and the progress bars follow it; Streamlit's blue info boxes were restyled (`stAlertContentInfo`); the button gradient was darkened for contrast. Lesson: a colour change touches more places than the CSS: SVG art, chart scales, native widgets and the Streamlit config all had blue in them, and only screenshots showed what was left.
