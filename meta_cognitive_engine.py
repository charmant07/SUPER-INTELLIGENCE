"""
meta_cognitive_engine.py

Upgraded MetaCognitiveEngine: a more powerful, extensible meta-cognitive layer that
combines symbolic graph reasoning, embeddings-based semantics, and optional
neural generation/scoring backends.

Key upgrades:
- Knowledge represented both as domain-keyed lists (backwards compatible) and a
  NetworkX graph for structural/pattern reasoning.
- Semantic embeddings via sentence-transformers (fallback to lightweight heuristics).
- Optional FAISS index for fast vector similarity if installed.
- Stronger analogy, pattern-transfer, concept-fusion, and constraint-relaxation
  algorithms combining graph metrics, embedding similarity, and lightweight
  optimization heuristics (scipy/sklearn where available).
- Robust dependency fallbacks and informative logging.
- Rich insight metadata (strategy, confidence, rationale).
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

# Base brain (provided in repo)
from cross_domain_brain import CrossDomainBrain

# Optional heavy libs — use gracefully if not installed
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
except Exception:
    cosine_similarity = None
    NearestNeighbors = None

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except Exception:
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


def _safe_cosine(a: List[float], b: List[float]) -> float:
    """Robust cosine similarity fallback if numpy/sklearn missing."""
    try:
        if np is None:
            # fallback: simple dot / (norms)
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb + 1e-12)
        a = np.array(a, dtype=float).reshape(1, -1)
        b = np.array(b, dtype=float).reshape(1, -1)
        if cosine_similarity is not None:
            return float(cosine_similarity(a, b)[0][0])
        # manual
        da = np.linalg.norm(a)
        db = np.linalg.norm(b)
        return float((a @ b.T) / (da * db + 1e-12))
    except Exception:
        return 0.0


class MetaCognitiveEngine(CrossDomainBrain):
    """
    Meta-cognitive engine that augments the CrossDomainBrain with heavy-duty
    semantic, graph, and optional neural capabilities for stronger insight generation.
    """

    def __init__(
        self,
        name: str = "PrometheusUltimate",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True,
        generation_model: Optional[str] = None,
    ):
        super().__init__(name)
        self.insight_history: List[Dict[str, Any]] = []
        self.connection_patterns_learned = 0

        # Graph representation for structural reasoning (optional)
        self.graph = nx.Graph() if nx is not None else None

        # Embedding model (sentence-transformers) if available
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        if SentenceTransformer is not None:
            try:
                logger.info("Loading SentenceTransformer model: %s", embedding_model_name)
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                logger.warning("Failed to load SentenceTransformer: %s", e)
                self.embedding_model = None

        # FAISS index for fast similarity search (optional)
        self.faiss_index = None
        self.faiss_vectors: List[List[float]] = []
        self.faiss_meta: List[Tuple[str, str]] = []  # (domain, concept)
        self.use_faiss = use_faiss and faiss is not None

        # Optional neural generator for concept fusion / explanation
        self.generation_model_name = generation_model
        self.generator = None
        if generation_model and pipeline is not None:
            try:
                logger.info("Loading generation pipeline: %s", generation_model)
                self.generator = pipeline("text-generation", model=generation_model, device=-1)
            except Exception as e:
                logger.warning("Failed to load generation pipeline: %s", e)
                self.generator = None

        # Local caches / helper data structures
        self._domain_centroids: Dict[str, List[float]] = {}
        self._last_rebuild = 0.0

    # ----------------------
    # Knowledge ingestion
    # ----------------------
    def ingest_knowledge(self, domain: str, concepts: List[str]) -> None:
        """
        Ingests concepts into both the underlying CrossDomainBrain (keeps
        backwards compatibility) and the graph/embedding indices.
        """
        # Keep base behavior (presumably stores in self.knowledge_base)
        try:
            super().ingest_knowledge(domain, concepts)
        except Exception:
            # If parent doesn't implement, ensure knowledge_base exists
            if not hasattr(self, "knowledge_base"):
                self.knowledge_base = {}
            self.knowledge_base.setdefault(domain, []).extend(concepts)

        # Add to graph (nodes labeled by domain)
        if self.graph is not None:
            for c in concepts:
                node_id = f"{domain}::{c}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id, domain=domain, concept=c, ingested_at=time.time()
                    )

        # Compute embeddings and add to FAISS/centroid
        embeddings = self._encode_concepts(concepts)
        if embeddings is not None:
            for concept, emb in zip(concepts, embeddings):
                self._faiss_add(domain, concept, emb)

        # Mark for centroid rebuild
        self._last_rebuild = 0.0

    # ----------------------
    # Embedding & FAISS helpers
    # ----------------------
    def _encode_concepts(self, concepts: List[str]) -> Optional[List[List[float]]]:
        """Return embeddings for list of concepts if model available."""
        if self.embedding_model is None:
            logger.debug("No embedding model available; skipping embeddings.")
            return None
        try:
            embs = self.embedding_model.encode(concepts, show_progress_bar=False)
            return [list(map(float, e)) for e in embs]
        except Exception as e:
            logger.warning("Embedding encoding failed: %s", e)
            return None

    def _faiss_add(self, domain: str, concept: str, emb: List[float]) -> None:
        """Add vector and metadata to local faiss structures (build index lazily)."""
        self.faiss_vectors.append(emb)
        self.faiss_meta.append((domain, concept))

        # Build or extend index lazily (rebuild only every so often)
        if self.use_faiss:
            try:
                d = len(emb)
                if self.faiss_index is None:
                    # simple IndexFlatL2 for demonstration; can be replaced with IVF/PCAs
                    self.faiss_index = faiss.IndexFlatIP(d)
                    np_arr = np.array(self.faiss_vectors).astype("float32")
                    # normalize for inner-product cosine-like search
                    norms = np.linalg.norm(np_arr, axis=1, keepdims=True) + 1e-12
                    np_arr = np_arr / norms
                    self.faiss_index.add(np_arr)
                else:
                    np_arr = np.array([emb]).astype("float32")
                    np_arr = np_arr / (np.linalg.norm(np_arr, axis=1, keepdims=True) + 1e-12)
                    self.faiss_index.add(np_arr)
            except Exception as e:
                logger.warning("FAISS add failed: %s", e)
                self.faiss_index = None

    def _rebuild_domain_centroids(self) -> None:
        """Compute per-domain centroids used for fast cross-domain analogies."""
        if self.embedding_model is None:
            return
        domain_vectors = {}
        for (domain, concept), vec in zip(self.faiss_meta, self.faiss_vectors):
            domain_vectors.setdefault(domain, []).append(vec)
        for d, vecs in domain_vectors.items():
            try:
                arr = np.array(vecs, dtype=float)
                centroid = list(np.mean(arr, axis=0))
                self._domain_centroids[d] = centroid
            except Exception:
                self._domain_centroids[d] = vecs[0] if vecs else []

        self._last_rebuild = time.time()

    # ----------------------
    # Insight generation pipeline
    # ----------------------
    def meta_cognitive_insight_generation(self, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Orchestrate multi-strategy insight generation and return top_k rich insight dicts.
        Each insight dict contains: insight, confidence, strategy, rationale, tags.
        """
        logger.info("META-COGNITIVE INSIGHT GENERATION ACTIVATED — using advanced pipelines")
        strategies = [
            ("analogy_detection", self._detect_analogies_across_domains, 1.2),
            ("pattern_transfer", self._transfer_patterns_between_domains, 1.0),
            ("concept_fusion", self._fuse_unrelated_concepts, 1.0),
            ("constraint_relaxation", self._relax_domain_constraints, 0.9),
        ]

        all_insights: List[Dict[str, Any]] = []
        for name, func, weight in strategies:
            try:
                logger.info("Running strategy: %s", name)
                results = func()
                # annotate confidence with strategy weight and time decay
                for r in results:
                    r["confidence"] = float(min(0.999, r.get("quality", 0.5) * weight))
                    r.setdefault("strategy", name)
                    all_insights.append(r)
            except Exception as e:
                logger.exception("Strategy %s failed: %s", name, e)

        # Rank and deduplicate (by insight text similarity)
        ranked = sorted(all_insights, key=lambda x: x.get("confidence", 0.0), reverse=True)
        deduped: List[Dict[str, Any]] = []
        seen_texts: List[str] = []
        for it in ranked:
            txt = it.get("insight", "")
            if any(_safe_cosine_text_similarity(txt, s) > 0.85 for s in seen_texts):
                continue
            deduped.append(it)
            seen_texts.append(txt)
            if len(deduped) >= top_k:
                break

        # Save history
        timestamp = time.time()
        for ins in deduped:
            ins_record = dict(ins)
            ins_record["generated_at"] = timestamp
            self.insight_history.append(ins_record)

        logger.info("Generated %d insights (returned top %d)", len(all_insights), len(deduped))
        return deduped

    # ----------------------
    # Strategy implementations
    # ----------------------
    def _detect_analogies_across_domains(self) -> List[Dict[str, Any]]:
        """
        Find robust analogical relationships using embeddings and graph structural
        motifs (e.g., high-degree nodes matching cross-domain centroids).
        """
        insights: List[Dict[str, Any]] = []
        domains = list(getattr(self, "knowledge_base", {}).keys())

        # Ensure centroids up-to-date
        if self.embedding_model is not None and (time.time() - self._last_rebuild > 60):
            self._rebuild_domain_centroids()

        # If embeddings available, compare centroids for domain-level analogies
        if self._domain_centroids:
            for i in range(min(len(domains), 6)):
                for j in range(i + 1, min(len(domains), 6)):
                    d1 = domains[i]
                    d2 = domains[j]
                    c1 = self._domain_centroids.get(d1)
                    c2 = self._domain_centroids.get(d2)
                    if c1 and c2:
                        sim = _safe_cosine(c1, c2)
                        quality = 0.2 + 0.8 * sim  # map to [0.2,1.0]
                        if quality > 0.45:
                            insight_text = f"Domain-level analogy: {d1} and {d2} share semantic structure (centroid similarity {sim:.2f})"
                            rationale = f"centroid_similarity={sim:.3f}"
                            insights.append({"insight": insight_text, "quality": quality, "rationale": rationale, "tags": ["domain_analogy"]})
                            logger.debug("Domain analogy: %s <-> %s (sim=%.3f)", d1, d2, sim)

        # Fine-grained concept-to-concept analogies using vector nearest neighbors
        if self.faiss_index is not None:
            # perform a few random probes to find cross-domain near neighbors
            n_probes = min(10, len(self.faiss_meta))
            indices = list(range(len(self.faiss_meta)))
            random.shuffle(indices)
            probes = indices[:n_probes]
            try:
                xb = np.array(self.faiss_vectors, dtype="float32")
                norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
                xb = xb / norms
                for idx in probes:
                    vec = xb[idx : idx + 1]
                    D, I = self.faiss_index.search(vec, k=6)
                    for score, i2 in zip(D[0], I[0]):
                        if i2 == -1 or i2 == idx:
                            continue
                        dmeta = self.faiss_meta[idx][0]
                        ometa = self.faiss_meta[i2][0]
                        if dmeta == ometa:
                            continue
                        quality = float(min(1.0, 0.2 + score))
                        insight_text = f"Analogy detected: {self.faiss_meta[idx][1]} ({dmeta}) ↔ {self.faiss_meta[i2][1]} ({ometa}) (sim={score:.3f})"
                        rationale = f"embedding_sim={score:.3f}"
                        insights.append({"insight": insight_text, "quality": quality, "rationale": rationale, "tags": ["analogy", "embedding"]})
            except Exception as e:
                logger.warning("Fine-grained analogy discovery failed: %s", e)

        # Graph-structural analogies (if graph available)
        if self.graph is not None:
            try:
                # compute degree patterns per domain: vector of sorted degrees
                domain_degree_vectors = {}
                for node, data in self.graph.nodes(data=True):
                    d = data.get("domain")
                    domain_degree_vectors.setdefault(d, []).append(self.graph.degree(node))
                for d, degs in domain_degree_vectors.items():
                    domain_degree_vectors[d] = sorted(degs, reverse=True)[:10]  # tail
                doms = list(domain_degree_vectors.keys())
                for i in range(len(doms)):
                    for j in range(i + 1, len(doms)):
                        v1 = domain_degree_vectors[doms[i]]
                        v2 = domain_degree_vectors[doms[j]]
                        # pad and compute simple correlation-like measure
                        L = max(len(v1), len(v2))
                        v1p = (v1 + [0] * L)[:L]
                        v2p = (v2 + [0] * L)[:L]
                        if np is not None:
                            corr = float(np.corrcoef(v1p, v2p)[0, 1]) if len(v1p) > 1 else 0.0
                        else:
                            corr = 0.0
                        quality = 0.3 + 0.6 * max(0.0, corr)
                        if quality > 0.5:
                            insight_text = f"Structural analogy: degree-patterns of {doms[i]} ≈ {doms[j]} (corr={corr:.2f})"
                            insights.append({"insight": insight_text, "quality": quality, "rationale": f"degree_corr={corr:.3f}", "tags": ["analogy", "graph"]})
            except Exception as e:
                logger.debug("Graph structural analogy step failed: %s", e)

        return insights

    def _transfer_patterns_between_domains(self) -> List[Dict[str, Any]]:
        """
        Transfer high-value structural or algorithmic patterns from one domain
        to another, using heuristic matching of hub motifs and embedding similarity.
        """
        insights: List[Dict[str, Any]] = []
        kb = getattr(self, "knowledge_base", {})
        domains = list(kb.keys())

        # Use historical known transfers (expanded and validated)
        known_transfers = [
            ("computer_science", "neural network", "biology", "neural networks"),
            ("mathematics", "graph theory", "engineering", "systems design"),
            ("physics", "thermodynamics", "engineering", "energy efficiency"),
        ]
        for from_d, from_term, to_d, to_term in known_transfers:
            if from_d in kb and to_d in kb:
                candidates_from = [c for c in kb[from_d] if from_term in str(c).lower()]
                candidates_to = [c for c in kb[to_d] if to_term in str(c).lower()]
                if candidates_from and candidates_to:
                    quality = random.uniform(0.6, 0.95)
                    insight = f"Pattern transfer: Apply '{candidates_from[0]}' ({from_d}) to improve '{candidates_to[0]}' ({to_d})"
                    rationale = f"historical_transfer: {from_term}->{to_term}"
                    insights.append({"insight": insight, "quality": quality, "rationale": rationale, "tags": ["transfer", "historical"]})

        # Graph motif transfer: find hubs in one domain and map to central nodes in another by embedding similarity
        if self.graph is not None and self.embedding_model is not None:
            try:
                domain_hubs = {}
                for node, data in self.graph.nodes(data=True):
                    d = data.get("domain")
                    domain_hubs.setdefault(d, []).append((node, self.graph.degree(node)))
                # pick top hubs in each domain
                hubs = {d: sorted(nodes, key=lambda x: x[1], reverse=True)[:3] for d, nodes in domain_hubs.items()}
                for d_from, hubs_from in hubs.items():
                    for d_to, hubs_to in hubs.items():
                        if d_from == d_to:
                            continue
                        for node_from, _ in hubs_from:
                            concept_from = self.graph.nodes[node_from]["concept"]
                            # compute embedding and find nearest concept in d_to using centroid or faiss meta
                            emb_from = self._encode_concepts([concept_from])
                            if not emb_from:
                                continue
                            emb_from = emb_from[0]
                            # approximate nearest by domain centroid
                            centroid = self._domain_centroids.get(d_to)
                            sim = _safe_cosine(emb_from, centroid) if centroid else 0.0
                            quality = 0.4 + 0.6 * sim
                            if quality > 0.55:
                                candidate_to = hubs_to[0][0]
                                concept_to = self.graph.nodes[candidate_to]["concept"]
                                insight = f"Motif transfer: Use '{concept_from}' pattern (hub in {d_from}) to accelerate '{concept_to}' (central in {d_to})"
                                rationale = f"hub_embedding_sim={sim:.3f}"
                                insights.append({"insight": insight, "quality": quality, "rationale": rationale, "tags": ["motif_transfer"]})
            except Exception as e:
                logger.debug("Motif transfer exception: %s", e)

        return insights

    def _fuse_unrelated_concepts(self) -> List[Dict[str, Any]]:
        """
        Fuse concepts via vector interpolation and optional generative "explain" step
        to craft human-readable fusion hypotheses.
        """
        insights: List[Dict[str, Any]] = []
        kb = getattr(self, "knowledge_base", {})
        domains = list(kb.keys())
        if len(domains) < 2:
            return insights

        # pick random domain pairs and attempt fusion
        attempts = 6
        for _ in range(attempts):
            d1, d2 = random.sample(domains, 2)
            c1 = random.choice(kb[d1])
            c2 = random.choice(kb[d2])

            # novelty and embedding gap
            emb_pair = self._encode_concepts([str(c1), str(c2)]) if self.embedding_model is not None else None
            if emb_pair:
                sim = _safe_cosine(emb_pair[0], emb_pair[1])
                novelty = 1.0 - sim
            else:
                novelty = 0.6 + random.uniform(0, 0.3)

            base_potential = 0.35 + 0.5 * novelty
            # give bonus if both are "fundamental" terms
            fundamentality_bonus = 0.0
            fundamental_terms = ["quantum", "evolution", "algorithm", "calculus", "system", "entropy"]
            if any(t in str(c1).lower() for t in fundamental_terms) and any(t in str(c2).lower() for t in fundamental_terms):
                fundamentality_bonus = 0.12
            potential = min(0.999, base_potential + fundamentality_bonus + random.uniform(0.05, 0.18))

            if potential > 0.5:
                # generate human text describing the fusion, prefer generator if available
                if self.generator is not None:
                    try:
                        prompt = f"Combine these two concepts into a concise research idea:\n1) {c1}\n2) {c2}\nProvide a 2-3 sentence hypothesis and one possible experiment or implementation."
                        gen = self.generator(prompt, max_length=180, num_return_sequences=1)
                        text = gen[0]["generated_text"].strip()
                    except Exception as e:
                        logger.debug("Generation failed: %s", e)
                        text = f"Concept fusion: Combining {c1} ({d1}) with {c2} ({d2}) may yield new paradigms."
                else:
                    text = f"Concept fusion: Combining {c1} ({d1}) with {c2} ({d2}) may yield new paradigms (potential {potential:.2f})."

                insights.append({"insight": text, "quality": float(potential), "rationale": f"novelty={novelty:.3f}", "tags": ["fusion", "creative"]})

        return insights

    def _relax_domain_constraints(self) -> List[Dict[str, Any]]:
        """
        Identify constraints in one domain that could be relaxed and propose cross-domain
        solutions. Uses simple optimization reasoning where scipy is available.
        """
        insights: List[Dict[str, Any]] = []
        kb = getattr(self, "knowledge_base", {})

        # Heuristic examples of constraints -> relaxations
        examples = [
            ("physics", "conservation laws", "biology"),
            ("biology", "evolutionary timescales", "engineering"),
            ("mathematics", "theoretical purity", "computer_science"),
        ]

        for domain, constraint, solver_domain in examples:
            if domain in kb and solver_domain in kb:
                # choose a candidate concept from both domains
                c_dom = random.choice(kb[domain])
                c_sol = random.choice(kb[solver_domain])
                # Use a heuristic quality: if embeddings available, check compatibility
                emb = self._encode_concepts([str(c_dom), str(c_sol)]) if self.embedding_model is not None else None
                compat = 0.5
                if emb:
                    compat = 0.2 + 0.8 * (1.0 - _safe_cosine(emb[0], emb[1]))  # more different -> potential for cross-relaxation
                # If scipy minimize available, perform a toy "relaxation" mapping
                if minimize is not None and emb:
                    try:
                        vec_dom = np.array(emb[0], dtype=float)
                        vec_sol = np.array(emb[1], dtype=float)

                        # toy objective: find alpha in [0,1] to minimize distance between relaxed_dom(alpha) and sol
                        def obj(alpha):
                            candidate = (1 - alpha[0]) * vec_dom + alpha[0] * vec_sol
                            return float(np.linalg.norm(candidate - vec_sol))

                        res = minimize(obj, x0=[0.5], bounds=[(0.0, 1.0)])
                        alpha = float(res.x[0]) if res.success else 0.5
                        quality = float(min(0.95, 0.45 + 0.5 * (1.0 - obj([alpha]) / (np.linalg.norm(vec_sol) + 1e-12))))
                        rationale = f"relax_alpha={alpha:.3f}"
                    except Exception as e:
                        logger.debug("Constraint relaxation minimize failed: %s", e)
                        quality = float(compat)
                        rationale = "compatibility_heuristic"
                else:
                    quality = float(compat)
                    rationale = "compatibility_heuristic"
                if quality > 0.45:
                    text = f"Constraint relaxation: By relaxing '{constraint}' in {domain}, applying '{c_sol}' from {solver_domain} could enhance '{c_dom}' (quality {quality:.2f})"
                    insights.append({"insight": text, "quality": quality, "rationale": rationale, "tags": ["constraint_relaxation"]})

        return insights


# ----------------------
# Utilities
# ----------------------
def _safe_cosine_text_similarity(a: str, b: str) -> float:
    """
    Lightweight text similarity fallback: short heuristics using token overlap
    and character-level similarity. If sentence-transformers is available, this
    can be replaced at runtime by computing embedding similarity.
    """
    try:
        if SentenceTransformer is not None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode([a, b], show_progress_bar=False)
            return float(_safe_cosine(list(emb[0]), list(emb[1])))
    except Exception:
        pass
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    # supplement with longest common substring normalized
    lcs = _longest_common_substring(a, b)
    lcs_score = len(lcs) / max(len(a), len(b), 1)
    return 0.6 * j + 0.4 * lcs_score


def _longest_common_substring(a: str, b: str) -> str:
    if not a or not b:
        return ""
    table = [[0] * (1 + len(b)) for _ in range(1 + len(a))]
    longest, x_longest = 0, 0
    for i in range(1, 1 + len(a)):
        for j in range(1, 1 + len(b)):
            if a[i - 1] == b[j - 1]:
                c = table[i - 1][j - 1] + 1
                table[i][j] = c
                if c > longest:
                    longest = c
                    x_longest = i
    return a[x_longest - longest : x_longest]


# ----------------------
# Lightweight manual test
# ----------------------
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    meta = MetaCognitiveEngine(generation_model=None)  # avoid heavy generator by default

    # Seed knowledge
    scientific_knowledge = {
        "physics": ["quantum entanglement", "relativity", "thermodynamics", "electromagnetism", "superconductivity"],
        "biology": ["evolution", "DNA replication", "cellular respiration", "neural networks", "enzyme catalysis"],
        "computer_science": ["algorithms", "machine learning", "neural network architectures", "optimization", "data structures"],
        "mathematics": ["calculus", "probability theory", "graph theory", "linear algebra", "complex systems"],
        "engineering": ["systems design", "materials science", "energy efficiency", "control theory", "signal processing"],
    }

    for domain, concepts in scientific_knowledge.items():
        meta.ingest_knowledge(domain, concepts)

    logger.info("Knowledge ingested. Domains: %s", list(meta.knowledge_base.keys()))
    insights = meta.meta_cognitive_insight_generation(top_k=8)
    logger.info("Top Insights:")
    for i, ins in enumerate(insights, 1):
        logger.info("%d) %s (conf=%.2f) [%s]", i, ins["insight"], ins["confidence"], ins.get("strategy"))
