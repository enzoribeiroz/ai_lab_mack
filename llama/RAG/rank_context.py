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
# Arquivo : llama/RAG/rank_context.py
# Síntese : Segundo estágio da recuperação. Recebe os candidatos devolvidos pela
#           busca MMR no Chroma e os reordena com um cross-encoder
#           (BAAI/bge-reranker-v2-m3), que pontua cada par (pergunta, trecho).
#           Apenas os top_n trechos mais relevantes seguem para o prompt do LLM.
#
# Histórico de alterações:
#   2026-02-26 - Guilherme Martins Silva - Versão inicial do re-ranker.
#   2026-02-26 - Guilherme Martins Silva - Comentários nos métodos da classe.
#   2026-11-15 - Enzo Ribeiro - Seleção automática de dispositivo (GPU/CPU),
#                precisão fp16 na GPU, parâmetro verbose e método score()
#                para os experimentos da N2; cabeçalho padronizado.
# =============================================================================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RankContext:

    def __init__(self, model, top_n=6, device=None, verbose=True):

        # Usa a GPU quando disponível; caso contrário, executa na CPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Carrega o tokenizer do modelo cross-encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Carrega o modelo de classificação (usado como reranker).
        # Na GPU usamos fp16, que reduz pela metade a memória sem perda relevante de qualidade.
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model, dtype=dtype)

        # Move o modelo para o dispositivo escolhido
        self.model.to(self.device)

        # Coloca o modelo em modo de inferência
        self.model.eval()

        # Número de documentos que serão mantidos após o reranking
        self.top_n = top_n

        # Controla a impressão do ranking (desligado nos experimentos em lote)
        self.verbose = verbose


    def score(self, query, texts, batch_size=8):
        """Devolve o score de relevância do cross-encoder para cada texto."""

        scores = []
        for i in range(0, len(texts), batch_size):
            # Cria pares (query, documento): o cross-encoder avalia relevância par a par
            pairs = [[query, t] for t in texts[i:i + batch_size]]

            # Tokeniza os pares para entrada no modelo
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # Inferência sem cálculo de gradiente (mais rápido)
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)

            scores.extend(logits.float().cpu().tolist())

        return scores


    def call_rank(self, query, documents):

        # Extrai apenas o texto dos documentos recuperados e pontua cada um
        texts = [d.page_content for d in documents]
        scores = self.score(query, texts)

        # Associa cada documento ao seu score
        scored_docs = list(zip(scores, documents))

        # Ordena do mais relevante para o menos relevante
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Seleciona apenas os top_n mais relevantes
        ranked = [d for _, d in scored_docs[:self.top_n]]

        # Log do ranking final
        if self.verbose:
            print("\n🏆 Reranker ranking:")
            for s, d in scored_docs[:self.top_n]:
                print(f"{s:.4f} | {d.metadata} | [doc_id: {d.id}]")

        # Retorna documentos rerankeados
        return ranked
