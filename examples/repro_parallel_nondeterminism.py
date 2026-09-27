"""Reproduce: order_invariant results vary between identical calls on a batching server.

Sends the *same* decision, with the *same* candidates in the *same* order, several times in a
row, once per `order_invariant_max_workers` setting, and reports how much the returned
probabilities move between those identical calls. Nothing about the input changes between
repeats, so any movement comes from the server, not from option order.

    python examples/repro_parallel_nondeterminism.py \
        --base-url http://127.0.0.1:8080/v1 --model <served-model-id>

    # llama.cpp only: also switch off the server's prompt cache for these requests
    python examples/repro_parallel_nondeterminism.py --model <id> --no-prompt-cache

Needs an OpenAI-compatible server that returns logprobs (llama.cpp, vLLM, SGLang).

Two separate sources show up, and the columns separate them:

* **Concurrent batching** (workers > 1). The per-candidate requests of one decision share
  batches whose composition depends on thread timing, and batched kernels are not
  bit-identical across batch compositions. Drift appears in *every* repeat, including the
  steady-state column, and dispatch order cannot fix it: the order is already identical on
  every repeat here.
* **Prompt-cache history** (some model/server pairs, even with workers = 1). A call's result
  depends on which prompt the server processed just before it, so the first call of a series
  differs and later identical calls agree. That shows as drift in "all repeats" but not in
  "repeats 2..N". `--no-prompt-cache` (llama.cpp's `cache_prompt: false`) removes it.

Read max |dp| as: ~1e-7 is float noise; 1e-3 and up is enough to flip a near tie. Label
flips need a near tie, so they depend on model and inputs; probability drift is the more
sensitive signal and shows up even when no label flips.
"""
import argparse
import json
import logging
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openjevpro.client import OpenJevProClient  # noqa: E402

CATEGORIES = {
    "card_arrival": "The card has been ordered but has not arrived, or the user asks where it is",
    "lost_or_stolen_card": "The card was lost or stolen and needs to be blocked",
    "change_pin": "Changing or resetting the card PIN",
    "card_payment_fee_charged": "A fee was charged on a card payment",
    "extra_charge_on_statement": "An unexpected or unrecognised charge appears on the statement",
    "transfer_fee_charged": "A fee was charged on a money transfer",
    "balance_not_updated": "A payment or transfer is not yet reflected in the balance",
}

# Deliberately includes ambiguous queries: label flips need a near tie.
QUERIES = [
    "How do I locate my card?",
    "Why is there a fee for an extra pound in my statement?",
    "I transferred money yesterday but my balance still shows the old amount.",
    "My card never came and I think someone may have taken it from my mailbox.",
    "There is a charge I don't recognise and I was also charged a fee for it.",
    "Can I change my PIN at an ATM, and is there a fee for that?",
    "I ordered a new card two weeks ago, is it lost?",
    "I paid by card abroad and now there's an extra amount on my account.",
]


def _spread(runs):
    """Largest change of any option's probability across a list of decisions."""
    return max(
        max(r.probabilities[k] for r in runs) - min(r.probabilities[k] for r in runs)
        for k in runs[0].probabilities
    )


def measure(client, repeats):
    """Identical calls per query. Returns counts, worst drift and per-query rows."""
    stats = {"drift": 0, "steady": 0, "flip": 0, "worst": 0.0, "worst_steady": 0.0}
    rows = []
    for query in QUERIES:
        runs = [
            client.decide_choice(
                state={"query": query},
                candidates=list(CATEGORIES),
                criteria=CATEGORIES,
                order_invariant=True,
            )
            for _ in range(repeats)
        ]
        dp_all, dp_steady = _spread(runs), _spread(runs[1:])
        winners = sorted({r.tentative_value or r.value for r in runs})
        stats["drift"] += dp_all > 0.0
        stats["steady"] += dp_steady > 0.0
        stats["flip"] += len(winners) > 1
        stats["worst"] = max(stats["worst"], dp_all)
        stats["worst_steady"] = max(stats["worst_steady"], dp_steady)
        rows.append((query, dp_all, dp_steady, winners))
    return stats, rows


def _disable_prompt_cache():
    """llama.cpp-specific: add `cache_prompt: false` to every request this process sends."""
    original = requests.post

    def post(url, *args, json=None, **kwargs):
        if isinstance(json, dict):
            json = dict(json, cache_prompt=False)
        return original(url, *args, json=json, **kwargs)

    requests.post = post


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=10, help="identical calls per query")
    ap.add_argument("--workers", default="1,8", help="comma-separated order_invariant_max_workers")
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}',
                    help='JSON passed as chat_template_kwargs; "{}" to send none')
    ap.add_argument("--no-prompt-cache", action="store_true",
                    help="llama.cpp only: send cache_prompt=false with every request")
    ap.add_argument("--verbose", action="store_true", help="per-query drift and winners")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    if args.no_prompt_cache:
        _disable_prompt_cache()

    kwargs = json.loads(args.chat_template_kwargs) or None
    n = len(QUERIES)
    print("model %s, %d queries x %d identical calls, same candidates in the same order, "
          "prompt cache %s\n" % (args.model, n, args.repeats,
                                 "off" if args.no_prompt_cache else "server default"))
    print("%-8s %14s %16s %8s %11s %14s" % ("workers", "drift (all)", "drift (2..N)", "flips",
                                            "max |dp|", "max |dp| 2..N"))
    print("-" * 76)
    for w in (int(x) for x in args.workers.split(",")):
        client = OpenJevProClient(base_url=args.base_url, model=args.model,
                                  chat_template_kwargs=kwargs, order_invariant_max_workers=w)
        client.decide_choice(state={"query": "warm up"}, candidates=list(CATEGORIES)[:2],
                             order_invariant=True)
        st, rows = measure(client, args.repeats)
        print("%-8d %14s %16s %8s %11.3g %14.3g" % (
            w, "%d/%d" % (st["drift"], n), "%d/%d" % (st["steady"], n), "%d/%d" % (st["flip"], n),
            st["worst"], st["worst_steady"]))
        if args.verbose:
            for query, dp_all, dp_steady, winners in rows:
                print("           |dp| %-9.3g 2..N %-9.3g %s  %s" % (dp_all, dp_steady, winners, query))

    print("\n~1e-7 is float noise; 1e-3 and up is enough to flip a near tie. Drift in 2..N means "
          "identical calls disagree in steady state (concurrency); drift only in 'all' means the "
          "first call differed (prompt-cache history).")


if __name__ == "__main__":
    main()
