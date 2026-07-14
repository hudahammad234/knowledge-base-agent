# Query Rewriting Approach — Member 2 (Retrieval)

## Why rewrite queries before retrieval?

Raw user questions are often too short, vague, or dependent on earlier
conversation turns to embed well. Example from the assignment spec:

> User: "Vacation" → System rewrites: "What are the company's annual leave
> and vacation policies?"

A one-word query like "Vacation" produces a weak embedding that can match
loosely-related chunks (e.g. anything mentioning "vacation" in passing)
instead of the actual policy section. Expanding it into a full question
before embedding significantly improves retrieval precision.

## How it works (`query_rewriter.py`)

1. **Input:** the raw user question, plus (optionally) the last N turns of
   conversation history from `memory.py`.
2. **No history case:** we send the raw question to Gemini with a system
   instruction asking it to expand the question into a complete,
   self-contained search query — without answering it.
3. **Follow-up case:** when conversation history exists, we include the last
   3 turns in the prompt so Gemini can resolve references such as "it",
   "that policy", or "what about maternity leave?" into a standalone query
   that carries the missing context (e.g. "What is the company's maternity
   leave policy?").
4. **Output contract:** Gemini is instructed to return *only* the rewritten
   query, nothing else — no preamble, no quotes, no explanation. This keeps
   the output directly usable as embedding input.
5. **Failure handling:** if the Gemini call fails for any reason (timeout,
   quota, malformed response), we log a warning and fall back to the
   original, unmodified question. This guarantees retrieval never breaks
   because of the rewriting step — worst case, retrieval quality degrades
   to "no rewriting," not a crash.

## Design decisions / trade-offs

| Decision | Reasoning |
|---|---|
| Use Gemini instead of rule-based expansion | Rule-based expansion (synonym lists, templates) doesn't generalize across arbitrary document domains. An LLM rewrite adapts to whatever documents are indexed. |
| Only last 3 turns of history included | Keeps the rewrite prompt small and cheap; older context is rarely needed to resolve a follow-up question and risks confusing the model with stale topics. |
| Low temperature (0.2) | Rewriting should be deterministic and literal, not creative — we want the same question rewritten the same way most of the time. |
| Always fall back to original query on failure | Retrieval must never hard-fail because of an upstream LLM call; a slightly worse query is better than no query. |
| Rewriting happens before embedding, not after | Embedding models capture full-question semantics far better than single keywords, so the rewrite must happen before, not after, the embedding step. |

## Example runs

| Raw query | Conversation history | Rewritten query |
|---|---|---|
| "Vacation" | none | "What are the company's annual leave and vacation policies?" |
| "what about maternity leave?" | Previous turn asked about annual leave | "What is the company's maternity leave policy?" |
| "How many sick days do I get?" | none | "How many sick days do I get?" *(already clear — returned unchanged)* |

## Known limitation

The rewriter currently doesn't detect when a follow-up question is
completely unrelated to prior turns (e.g. topic switch). In that case it
could over-include stale context. This is listed as a possible improvement
if we tackle the "simple AI agent" bonus challenge, which would first decide
whether a question needs retrieval/rewriting at all.
