# =============================================================================
# Projeto : Agente Especialista em GRC - RAG + LLM local sobre o COSO ERM 2017
# Curso   : Sistemas de Informação - Inteligência Artificial (7ºK, noite)
# Prof.   : Dr. Leandro Zerbinatti
# Inst.   : Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática
#
# Integrantes:
#   Guilherme Martins Silva - RA 10417140 - 10417140@mackenzista.com.br
#   Enzo Ribeiro            - RA 10418262 - 10418262@mackenzista.com.br
#   Pedro Nogueira Ribeiro  - RA 10324924 - 10324924@mackenzista.com.br
#
# Arquivo : llama/baselines.py
# Síntese : Recuperadores usados na comparação experimental da N2. Todos operam
#           sobre os MESMOS chunks do banco Chroma e devolvem a lista de textos
#           no formato esperado pelo RAGEvaluator (pipeline_fn(query, k)):
#             - aleatório (piso de referência da métrica);
#             - BM25 (baseline lexical prometido na N1);
#             - denso puro (similaridade de cosseno bge-m3, top-n);
#             - MMR sem re-ranking;
#             - denso + re-ranking (sem MMR);
#             - MMR + re-ranking (pipeline do agente).
#
# Histórico de alterações:
#   2026-11-15 - Enzo Ribeiro - Versão inicial.
# =============================================================================
import random
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-zà-ÿ0-9]+")

# Stopwords mínimas (EN + PT) para o BM25 não pontuar palavras vazias
STOPWORDS = set("""
a an and are as at be by for from has have how in is it its of on or that the this to was what
which with does do should can between beyond across according about into than other their
o os as um uma uns umas de do da dos das e é em no na nos nas por para com que qual quais
quantos quantas como se ao aos à às ou ser deve ele ela seu sua seus suas sao são há
""".split())


def tokenize(text: str):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def load_chunks(chroma):
    """Lê todos os chunks persistidos no Chroma como Documents (com id e metadados)."""
    data = chroma.get(include=["documents", "metadatas"])
    return [Document(page_content=t, metadata=m or {}, id=i)
            for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])]


def random_retriever(chunks, top_n=6, seed=42):
    rng = random.Random(seed)

    def pipeline(query, k=None):
        return [d.page_content for d in rng.sample(chunks, top_n)]
    return pipeline


def bm25_retriever(chunks, top_n=6, query_map=None):
    """BM25 Okapi. query_map permite trocar a pergunta (ex.: PT -> tradução EN)."""
    bm25 = BM25Okapi([tokenize(d.page_content) for d in chunks])

    def pipeline(query, k=None):
        q = query_map.get(query, query) if query_map else query
        scores = bm25.get_scores(tokenize(q))
        best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [chunks[i].page_content for i in best]
    return pipeline


def dense_retriever(chroma, top_n=6):
    def pipeline(query, k=None):
        return [d.page_content for d in chroma.similarity_search(query, k=top_n)]
    return pipeline


def mmr_retriever(chroma, top_n=6, fetch_k=50, lambda_mult=0.5):
    def pipeline(query, k=None):
        docs = chroma.max_marginal_relevance_search(query, k=top_n, fetch_k=fetch_k, lambda_mult=lambda_mult)
        return [d.page_content for d in docs]
    return pipeline


def dense_rerank_retriever(chroma, reranker, k=20):
    def pipeline(query, _k=None):
        docs = chroma.similarity_search(query, k=k)
        return [d.page_content for d in reranker.call_rank(query, docs)]
    return pipeline


def mmr_rerank_docs(chroma, reranker, query, k=20, fetch_k=50, lambda_mult=0.5):
    """Pipeline do agente (mesmos parâmetros de CallLlama._extract_info_chunks)."""
    docs = chroma.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    return reranker.call_rank(query, docs)


def mmr_rerank_retriever(chroma, reranker):
    def pipeline(query, k=20):
        return [d.page_content for d in mmr_rerank_docs(chroma, reranker, query, k=k)]
    return pipeline
