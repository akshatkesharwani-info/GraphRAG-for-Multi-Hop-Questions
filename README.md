# GraphRAG for Multi-Hop Questions

Normal RAG finds text that looks similar to the question. That breaks on **multi-hop questions**, where the answer needs several connected facts ("Which city is the company that acquired the startup founded by Riya Sharma headquartered in?"). This project builds a knowledge graph from text and walks it, then tests that idea honestly against simpler baselines.

Built in Google Colab with Groq (`openai/gpt-oss-120b`), NetworkX and sentence-transformers.

## What it does

1. Takes 14 short passages about a **made-up** set of companies and people, so the model cannot answer from memory.
2. Uses the LLM to extract (subject, relation, object) triples from each passage and builds a graph.
3. **Checks the extraction** against a hand-written list of facts (recall and precision) and prints every relation for a human to read.
4. Answers 10 questions (3 need 1 hop, 3 need 2 hops, 4 need 3 hops) with four methods:
   - **Vector RAG:** the 3 most similar passages
   - **All passages in the prompt:** the whole corpus (an upper bound on a tiny corpus)
   - **Graph, 3 hops fixed:** find the entities in the question and send every fact within 3 hops
   - **Graph, adaptive hops:** start with 1 hop and only widen to 2 or 3 if the model says it needs more facts
5. Finds communities in the graph and writes a one-sentence summary of each, to answer a big-picture question.

## Results from the run

| Method | Accuracy | Avg words sent to the LLM | Avg LLM calls |
|---|---|---|---|
| Vector RAG (top 3) | 30% | 30 | 1.0 |
| All passages in the prompt | 100% | 143 | 1.0 |
| Graph, 3 hops fixed | 100% | 84 | 1.0 |
| Graph, adaptive hops | 100% | 56 | 2.1 |

Accuracy by hops needed (vector RAG): 100% on 1-hop questions, **0% on 2-hop and 3-hop questions**. All three graph and full-context methods scored 100% at every level.

- The adaptive method used exactly as many hops as each question needed (1, 2 and 3).
- Triple extraction: 21 triples, 18 nodes, 100% recall and 100% precision against the hand-written list.
- Vector RAG mostly answered "unknown", but on one 3-hop question it confidently gave a wrong answer (Jaipur instead of Pune).
- The graph split into 6 communities, and the summaries answered "what are the main groups of companies and people?" correctly.

## What the evaluation showed

- **Vector search fails exactly where the question needs a chain of facts.** The top 3 passages rarely contain every link in the chain.
- **GraphRAG did not beat putting everything in the prompt.** On 14 passages, the full corpus fits easily and scored 100% too. The graph's real advantage here was context size: the adaptive graph sent about 61% fewer words than the full corpus (56 vs 143), at the cost of more LLM calls. That advantage matters when the corpus is too big for one prompt.
- **Extraction quality decides everything.** An earlier version stored "Vikram Rao studied at IIT Indore before joining Orion" as `previously_worked_at`. The tighter extraction prompt and the check against hand-written facts fixed that.

## Limitations

- **Tiny and made-up:** 14 passages, 10 questions. Treat the numbers as a demo, not a benchmark.
- **Entity linking is plain string matching**, so it only finds entities named exactly as in the graph.
- **The adaptive method trusts the model** to say "NEED MORE FACTS" instead of guessing.
- Next test: a few hundred real paragraphs, where the full corpus no longer fits in a prompt.

## Tech stack

NetworkX, sentence-transformers (`all-MiniLM-L6-v2`), Groq API, pandas, matplotlib.

## How to run

1. Open the notebook in Google Colab.
2. Run the cells from top to bottom.
3. When asked, paste your own Groq API key (free at console.groq.com). It is hidden while typing and is never saved in the notebook.

It makes about 85 API calls.

## Files the notebook creates

- `graphrag_triples.csv`: the extracted knowledge graph
- `graphrag_vs_vector_results.csv`: every answer from every method

---

Built by **Akshat Kesharwani** | [GitHub](https://github.com/akshatkesharwani-info) | [Portfolio](https://akshatkesharwani-info.github.io/)
