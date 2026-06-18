"""Hard track: pragmatic intents over a shared-topic corpus.

The easy track (``dataset.py``) tests *topical* ambiguity ("python" the
snake vs. the language), which a diagonal lens solves by gating topic
dimensions, hence the near-ceiling scores. This track is harder: for each
technical topic there are four documents, a *tutorial*, an *api
reference*, a *troubleshooting* note, and a *conceptual* explanation,
that all share the topic's vocabulary. The query names the topic; only the
intent (the pragmatic flavor) can pick the right document. The flavor is
carried implicitly by the prose, not by a "Tutorial:" label, so separating
the four requires understanding *purpose*, not keywords.

This leaves headroom (the full stack does not saturate) and is the kind of
case where a diagonal lens may fall short of a low-rank one, leaving room
to evaluate richer per-intent adapters.

- ``DOCS`` / ``INTENTS`` / ``CASES``: the held-out evaluation set.
- ``TRAIN_CASES``: a disjoint set of (query -> relevant doc) pairs per
  intent, for fitting feedback-driven models (the ladder). Eval queries
  are bare topic phrases; training queries carry the flavor explicitly.
"""

from __future__ import annotations

DOC_TYPES = ("tutorial", "reference", "troubleshooting", "concept")

# topic_key -> {"label", "query", and one passage per DOC_TYPE}
GRID: dict[str, dict[str, str]] = {
    "groupby": {
        "label": "groupby",
        "query": "dataframe groupby aggregation",
        "tutorial": "Start by calling groupby on the column you want to aggregate over, then chain an aggregation like sum or mean, and finally reset_index to get a flat frame back, try it on a small sample first.",
        "reference": "DataFrame.groupby(by, axis=0, level=None, as_index=True, sort=True) returns a GroupBy object; aggregate with .agg, .sum, .mean, or .apply, and pass as_index=False to keep the grouping columns.",
        "troubleshooting": "If groupby drops rows, NaN keys are excluded by default, pass dropna=False; if the result is unexpectedly a Series instead of a frame, you selected a single column before aggregating.",
        "concept": "groupby follows a split-apply-combine model: rows are partitioned by key, a function runs per partition, and the pieces are stitched back together, which is why such aggregations vectorize well.",
    },
    "async": {
        "label": "async/await",
        "query": "async await coroutines",
        "tutorial": "Mark a function async def, await the coroutines inside it, and run the top-level one with asyncio.run; convert blocking calls to their async equivalents one at a time as you go.",
        "reference": "async def defines a coroutine; await suspends until an awaitable resolves; asyncio.gather(*aws, return_exceptions=False) schedules many concurrently and asyncio.create_task schedules one.",
        "troubleshooting": "A 'coroutine was never awaited' warning means you called it without await; if the event loop is already running, asyncio.run raises, so await the coroutine or wrap it in a task instead.",
        "concept": "async and await are cooperative concurrency on a single thread: awaiting yields control to the event loop, which runs other tasks until I/O is ready, so it scales I/O-bound work rather than CPU-bound work.",
    },
    "regex": {
        "label": "regular expressions",
        "query": "regular expression matching",
        "tutorial": "Begin with literal characters, add a character class such as a digit class, then group with parentheses and repeat with quantifiers, build the pattern up one piece at a time in a tester.",
        "reference": "re.compile(pattern, flags) returns a Pattern; .match anchors at the start, .search scans anywhere, .findall returns all non-overlapping matches, and captured groups are referenced by number or by name.",
        "troubleshooting": "A catastrophically slow regular expression usually comes from nested quantifiers that backtrack exponentially; flatten the pattern or make the quantifier possessive to avoid the blowup.",
        "concept": "A regular expression denotes a set of strings recognized by a finite automaton; the engine walks states as it consumes characters, which explains both its speed and its backtracking pitfalls.",
    },
    "compose": {
        "label": "docker compose",
        "query": "docker compose services",
        "tutorial": "Write a compose file listing each service and its image, declare the ports and volumes, then bring the whole stack up with one up command, add services one at a time and test as you go.",
        "reference": "A compose file has a services map; each service takes image, build, ports as host:container, volumes, environment, and depends_on; the up command runs the stack and down tears it all back down.",
        "troubleshooting": "If one service cannot reach another, use the service name as the hostname, not localhost; if a port is already allocated, another process holds it, so change the host side of the port mapping.",
        "concept": "Compose declares a multi-container application as a single file: it creates a shared network so services find each other by name, modeling the deployment as one declarative unit instead of ad-hoc run commands.",
    },
    "gitmerge": {
        "label": "git merge conflicts",
        "query": "git merge conflict",
        "tutorial": "When a merge stops with a conflict, open each marked file, choose the lines you want between the conflict markers, delete the markers, then add the file and commit to complete the merge.",
        "reference": "The merge command joins two histories; on a conflict the working tree gains HEAD and incoming markers around the overlapping hunk; abort restores the pre-merge state and the diff shows the conflicting regions.",
        "troubleshooting": "If you keep resolving the same conflicts, the rerere feature can record and replay your resolutions; if you merged the wrong branch, abort backs the merge out before you commit it.",
        "concept": "A merge conflict arises when two branches change the same lines and the tool cannot pick a winner; it falls back to a three-way comparison against the common ancestor and asks you to resolve the overlap.",
    },
    "jwt": {
        "label": "JWT authentication",
        "query": "jwt authentication token",
        "tutorial": "Issue a token on login by signing a payload with your secret, send it to the client, then on each request read the bearer header and verify the signature before trusting any of the claims.",
        "reference": "A token is header, payload, and signature, base64url-encoded; sign with a shared secret or a private key; standard claims include expiry, issued-at, issuer, and subject; verification checks signature and expiry.",
        "troubleshooting": "An invalid-signature error usually means the verifying key differs from the signing one; a token that never expires is missing an expiry claim, and clock skew between servers can reject otherwise fresh tokens.",
        "concept": "A signed token is a stateless, self-contained credential: because the server verifies it by signature alone, no session store is needed, trading easy revocation for horizontal scalability.",
    },
    "flexbox": {
        "label": "CSS flexbox",
        "query": "css flexbox layout",
        "tutorial": "Set the container to display flex, choose a direction, then align the children with the main-axis and cross-axis alignment properties, change one property at a time to watch the effect.",
        "reference": "A flex container takes flex-direction, flex-wrap, and the two alignment properties; items take grow, shrink, and basis (the flex shorthand) plus a self-alignment override on the cross axis.",
        "troubleshooting": "If items overflow instead of wrapping, enable wrapping; if main-axis alignment seems to do nothing, your main axis runs the other way, the direction property decides which axis alignment affects.",
        "concept": "Flexbox lays items along a single main axis with a perpendicular cross axis; free space is distributed by growing and shrinking items, which is why it excels at one-dimensional, content-sized layouts.",
    },
    "sqljoin": {
        "label": "SQL joins",
        "query": "sql join tables",
        "tutorial": "Start from your primary table, add a join to the related table, write the matching condition on the keys, then select the columns you need, run it and check the row count before adding more joins.",
        "reference": "An inner join returns matching rows; a left join keeps all left rows with nulls for non-matches; the on clause states the predicate, a cross join is the Cartesian product, and using shortcuts equal-named keys.",
        "troubleshooting": "If a join multiplies your rows, the key is not unique on one side; if expected rows vanish, an inner join dropped the non-matches, so switch to a left join or check for null keys.",
        "concept": "A join combines rows from two tables by a predicate: conceptually it forms the Cartesian product and filters it, though the planner uses hash or merge strategies so it never materializes the whole product.",
    },
    "httpcache": {
        "label": "HTTP caching",
        "query": "http caching headers",
        "tutorial": "Add a cache-control header to your responses, set a max-age for how long they stay fresh, then add an entity tag so clients can revalidate and get a cheap not-modified, begin with static assets.",
        "reference": "Cache-control directives include max-age, no-cache, no-store, and private or public; an entity tag with the conditional request header enables a 304 that skips the body; expires is the older absolute form.",
        "troubleshooting": "If clients serve stale content, the max-age is too long or no-cache is missing; if nothing caches at all, a set-cookie or a private directive may be suppressing the shared caches.",
        "concept": "HTTP caching trades freshness for latency: responses carry their own freshness lifetime so browsers and intermediaries can reuse them, and revalidation lets a cache confirm a stored copy without refetching it.",
    },
    "pytestfix": {
        "label": "pytest fixtures",
        "query": "pytest fixtures",
        "tutorial": "Write a function, decorate it as a fixture, and request it by adding its name as a test argument; return the value the test needs, and use yield instead of return when you also need teardown.",
        "reference": "The fixture decorator takes a scope of function, module, or session and an optional params list; request a fixture by parameter name; yield separates setup from teardown; a conftest shares fixtures across files.",
        "troubleshooting": "A 'fixture not found' error means it is out of scope, so move it to a conftest; if setup runs too often, widen the scope; a mutable session fixture shared between tests causes flaky ordering bugs.",
        "concept": "Fixtures invert setup: instead of each test building its own state, tests declare what they need and the framework injects it, which makes dependencies explicit and lets the framework manage lifetime and reuse.",
    },
    "kafka": {
        "label": "Kafka consumers",
        "query": "kafka consumer group",
        "tutorial": "Create a consumer with a group id, subscribe to a topic, then poll in a loop and process each record; commit the offsets after handling a batch so you resume where you left off.",
        "reference": "A consumer is configured with the broker list, a group id, and an offset-reset policy; polling fetches records, synchronous or asynchronous commits store offsets, and partitions are balanced across the group.",
        "troubleshooting": "If messages are reprocessed, you committed offsets before finishing the work or a rebalance reset them; if a consumer stalls, the max poll interval expired because processing a batch took too long.",
        "concept": "A consumer group splits a topic's partitions among its members so each partition is read by exactly one consumer, which is how the system scales reads horizontally while preserving per-partition order.",
    },
    "reacthooks": {
        "label": "React hooks",
        "query": "react hooks state",
        "tutorial": "Call the state hook at the top of your component to hold a value, then the effect hook to run side effects after render; keep hooks at the top level and update through the setter to trigger a re-render.",
        "reference": "The state hook returns a value and a setter; the effect hook runs after render when its dependency array changes and its returned function is the cleanup; the memo hooks cache values and callbacks by dependencies.",
        "troubleshooting": "An infinite render loop usually means you set state inside an effect without a correct dependency array; a stale value in a callback comes from a missing dependency closing over old state.",
        "concept": "Hooks let function components hold state and effects by relying on a stable call order across renders, which is why they must run unconditionally at the top level rather than inside conditions or loops.",
    },
}

INTENTS: list[tuple[str, str, list[str]]] = [
    ("tutorial", "a step-by-step, hands-on walkthrough for getting started and following along", [
        "a step by step walkthrough", "how do I get started", "follow along beginner guide",
        "first do this then that",
    ]),
    ("reference", "the precise API: signatures, parameters, options, and what it returns", [
        "the exact signature and parameters", "the api and its arguments",
        "what it returns and the options", "the precise specification",
    ]),
    ("troubleshooting", "diagnosing a failure: the error, what causes it, and the fix", [
        "why is this failing and how to fix it", "a common error and its resolution",
        "diagnosing the bug", "what causes this and the workaround",
    ]),
    ("concept", "the mental model: what it is, why it works, and how it works underneath", [
        "what it is and why it works", "the mental model and rationale",
        "how it works under the hood", "the idea behind it",
    ]),
]

# training-query templates per intent (flavor explicit; disjoint from the
# bare-topic eval queries, so they form a real held-out training signal)
_TRAIN_TEMPLATES: dict[str, list[str]] = {
    "tutorial": ["how do I get started with {label}", "a step by step {label} walkthrough"],
    "reference": ["{label} parameters and signature", "the {label} api options and return value"],
    "troubleshooting": ["{label} not working how do I fix it", "a common {label} error and its cause"],
    "concept": ["how does {label} actually work", "why does {label} work the way it does"],
}

DOCS: list[tuple[str, str]] = [
    (f"{topic}-{dtype}", entry[dtype])
    for topic, entry in GRID.items()
    for dtype in DOC_TYPES
]

CASES: list[tuple[str, str, str]] = [
    (entry["query"], dtype, f"{topic}-{dtype}")
    for topic, entry in GRID.items()
    for dtype in DOC_TYPES
]

TRAIN_CASES: list[tuple[str, str, str]] = [
    (tmpl.format(label=entry["label"]), dtype, f"{topic}-{dtype}")
    for topic, entry in GRID.items()
    for dtype in DOC_TYPES
    for tmpl in _TRAIN_TEMPLATES[dtype]
]
