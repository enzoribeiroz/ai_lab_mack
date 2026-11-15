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
# Arquivo : llama/EvaluationRAG.py
# Síntese : Avaliação do recuperador (retriever). Para cada pergunta do conjunto
#           de avaliação, executa o pipeline e verifica, por similaridade de
#           cosseno (embeddings BAAI/bge-m3), se algum trecho devolvido
#           corresponde a um trecho de referência. Calcula Hit Rate, MRR e
#           Precision@1 em uma varredura de thresholds de similaridade.
#
# Histórico de alterações:
#   2026-02-26 - Guilherme Martins Silva - Versão inicial do RAGEvaluator.
#   2026-11-15 - Enzo Ribeiro - Cache dos documentos recuperados e das
#                similaridades por pergunta (antes o pipeline era executado de
#                novo para cada threshold; o resultado é idêntico e ~25x mais
#                rápido); método per_query() com o detalhamento por pergunta;
#                parâmetro device (avaliador na CPU em GPUs de 6 GB);
#                cabeçalho padronizado.
# =============================================================================
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Callable
import torch.nn.functional as F
from dataclasses import dataclass
import pandas as pd

@dataclass
class EvaluationSample:
    query: str
    relevant_chunks: List[str]

class RAGEvaluator:

    def __init__(
        self,
        pipeline_fn: Callable[[str, int], List[str]],
        dataset: List[EvaluationSample],
        model_name: str = "BAAI/bge-m3",
        device: str = None
    ):
        self.pipeline_fn = pipeline_fn
        self.dataset = dataset
        # device="cpu" libera VRAM quando o índice e o re-ranker já ocupam a GPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Carregando via Hugging Face Transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Cache: k -> lista (uma por pergunta) com a maior similaridade de cada
        # documento recuperado contra os trechos de referência
        self._cache = {}

    def _get_embeddings(self, texts: List[str]):
        """Gera embeddings normalizados (token [CLS] do bge-m3) para uma lista de textos."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            # Normalização para que o produto escalar seja igual à similaridade de cosseno
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _semantic_check(self, retrieved: List[str], relevant: List[str], threshold: float) -> int:
        if not retrieved or not relevant:
            return 0

        emb_retrieved = self._get_embeddings(retrieved)
        emb_relevant = self._get_embeddings(relevant)

        cos_sim_matrix = torch.mm(emb_retrieved, emb_relevant.t())

        max_sim_per_relevant, _ = torch.max(cos_sim_matrix, dim=0)

        # conta quantos relevantes foram cobertos
        hits_count = torch.sum(max_sim_per_relevant >= threshold).item()

        return hits_count

    def _doc_similarities(self, k: int):
        """Executa o pipeline uma única vez por pergunta e guarda, para cada documento
        recuperado (na ordem do ranking), a maior similaridade com os trechos de referência."""
        if k in self._cache:
            return self._cache[k]

        sims_per_query = []
        for sample in self.dataset:
            retrieved = self.pipeline_fn(sample.query, k)
            if not retrieved or not sample.relevant_chunks:
                sims_per_query.append([])
                continue
            emb_retrieved = self._get_embeddings(retrieved)
            emb_relevant = self._get_embeddings(sample.relevant_chunks)
            # documento relevante <=> similaridade máxima com algum trecho de referência >= threshold
            max_sim_per_doc = torch.mm(emb_retrieved, emb_relevant.t()).max(dim=1).values
            sims_per_query.append(max_sim_per_doc.cpu().tolist())

        self._cache[k] = sims_per_query
        return sims_per_query

    @staticmethod
    def _first_hit_rank(sims: List[float], threshold: float):
        for rank, s in enumerate(sims, start=1):
            if s >= threshold:
                return rank
        return None

    def create_evaluation_df(self):
        # Definimos as colunas exatas para as novas métricas
        columns = [
            "N° Interação",
            "similarity_threshold",
            "hit_rate",
            "mrr",
            "precision_at_1",
            "N° Hits",
            "N° Miss"
        ]
        return pd.DataFrame(columns=columns)

    def per_query(self, k: int, similarity_threshold: float) -> pd.DataFrame:
        """Detalhamento por pergunta: posição do primeiro acerto e similaridade do top-1."""
        rows = []
        for sample, sims in zip(self.dataset, self._doc_similarities(k)):
            rank = self._first_hit_rank(sims, similarity_threshold)
            rows.append({
                "query": sample.query,
                "first_hit_rank": rank,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "top1_similarity": sims[0] if sims else None,
                "best_similarity": max(sims) if sims else None,
            })
        return pd.DataFrame(rows)

    def evaluate(self, k: int, similarity_threshold: float, max_threshold=0.99, verbose=True):
        rows = []
        sims_per_query = self._doc_similarities(k)
        n = len(self.dataset)

        # Define o número de passos (0.01 em 0.01)
        steps = int(round((max_threshold - similarity_threshold) * 100))

        for i in range(steps + 1):
            thr = round(similarity_threshold + 0.01 * i, 2)
            running_hits = 0        # Para Hit Rate
            running_mrr = 0.0       # Para Mean Reciprocal Rank
            running_p1 = 0.0        # Para Precision@1

            for sims in sims_per_query:
                first_hit_rank = self._first_hit_rank(sims, thr)

                # Hit Rate: o documento relevante apareceu em qualquer lugar da lista?
                if first_hit_rank is not None:
                    running_hits += 1
                    # MRR: 1 / posição do primeiro acerto (Ex: 1º=1.0, 2º=0.5, 3º=0.33)
                    running_mrr += 1.0 / first_hit_rank
                    # Precision@1: o primeiro da lista é relevante?
                    if first_hit_rank == 1:
                        running_p1 += 1.0

            rows.append({
                "N° Interação": i,
                "similarity_threshold": thr,
                "hit_rate": running_hits / n,
                "mrr": running_mrr / n,
                "precision_at_1": running_p1 / n,
                "N° Hits": running_hits,
                "N° Miss": n - running_hits
            })

            if verbose:
                print(f"Thres: {thr:.2f} | HitRate: {running_hits / n:.3f} | MRR: {running_mrr / n:.3f} | P@1: {running_p1 / n:.3f}")

        return pd.DataFrame(rows)
