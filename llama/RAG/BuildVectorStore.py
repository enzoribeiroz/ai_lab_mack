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
# Arquivo : llama/RAG/BuildVectorStore.py
# Síntese : Pipeline de ingestão da base de conhecimento. Carrega os .txt do
#           corpus, segmenta por seções, divide em chunks por tokens (tokenizer
#           do BAAI/bge-m3), remove duplicatas (hash SHA-256 e similaridade
#           semântica > 0,95) e persiste os embeddings em um banco Chroma.
#           Também expõe build_chunks() para os experimentos de tamanho de chunk.
#
# Histórico de alterações:
#   2026-02-26 - Guilherme Martins Silva - Versão inicial do pipeline de ingestão.
#   2026-02-26 - Guilherme Martins Silva - Comentários da função de normalização.
#   2026-11-15 - Enzo Ribeiro - Caminhos padrão relativos ao módulo, seleção
#                automática de dispositivo, chunk_size/chunk_overlap/cleaner
#                configuráveis e método build_chunks() (experimentos da N2);
#                cabeçalho padronizado.
# =============================================================================
import os
import logging
import hashlib
import re
import unicodedata
import numpy as np
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RAG")


class PDFVectorStore:

    def __init__(
        self,
        docs_path=None,
        persist_dir=None,
        embedding_model="BAAI/bge-m3",
        chunk_size=400,
        chunk_overlap=80,
        text_cleaner=None,
        device=None
    ):

        # Caminhos padrão relativos a esta pasta (llama/RAG), independentes do cwd
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.docs_path = docs_path or os.path.join(base_dir, "docs_uploader", "documents_txt")
        self.persist_dir = persist_dir or os.path.join(base_dir, "db")

        # Parâmetros de segmentação (padrão = configuração usada no índice persistido)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Função opcional de limpeza aplicada ao texto bruto antes da segmentação
        self.text_cleaner = text_cleaner

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info("🚀 Carregando embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 8}  # lote pequeno: cabe em GPUs de 6 GB
        )

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)

        self.db = None


    # ==================================================
    # INIT
    # ==================================================

    def initialize(self):

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            log.info("📦 DB já existe → carregando...")
            self.db = self._load_db()
        else:
            log.info("🆕 DB não existe → construindo índice...")
            self.db = self._build_db()

        return self.db


    def get_db(self):
        if self.db is None:
            self.initialize()
        return self.db


    # ==================================================
    # BUILD
    # ==================================================

    def build_chunks(self):
        """Executa todo o pré-processamento e devolve os chunks finais (sem persistir)."""

        docs = self._load_files()
        docs = self._split_by_sections(docs)

        chunks = self._chunk(docs)

        chunks = self._hash_dedup(chunks)
        chunks = self._semantic_dedup(chunks)
        return chunks


    def _build_db(self):

        chunks = self.build_chunks()

        log.info("🧠 Gerando embeddings + salvando Chroma...")

        self.db = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        log.info("✅ DB persistido!")
        return self.db


    def _load_db(self):
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )


    # ==================================================
    # LOAD TXT
    # ==================================================

    def _load_files(self):

        docs = []

        for file in os.listdir(self.docs_path):

            if not file.endswith(".txt"):
                continue

            path = os.path.join(self.docs_path, file)

            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()

            for i, d in enumerate(loaded):
                if self.text_cleaner is not None:
                    d.page_content = self.text_cleaner(d.page_content)
                d.metadata["source"] = file
                d.metadata["doc_id"] = f"{file}_{i}"

            docs.extend(loaded)

        log.info(f"📄 documentos carregados: {len(docs)}")
        return docs


    # ==================================================
    # SPLIT POR SEÇÕES (TXT INTELIGENTE)
    # ==================================================

    def _split_by_sections(self, docs):

        new_docs = []

        section_pattern = r"(CAP[IÍ]TULO.*|PRINC[IÍ]PIO.*|[0-9]+\.[0-9]+.*)"

        for d in docs:

            sections = re.split(section_pattern, d.page_content)

            buffer = ""
            section_title = None

            for part in sections:

                if re.match(section_pattern, part):

                    if buffer:
                        new_docs.append(Document(
                            page_content=buffer,
                            metadata=d.metadata
                        ))
                        buffer = ""

                    section_title = part
                    buffer += part + "\n"

                else:
                    buffer += part

            if buffer:
                new_docs.append(Document(
                    page_content=buffer,
                    metadata=d.metadata
                ))

        log.info(f"🧱 seções detectadas: {len(new_docs)}")
        return new_docs


    # ==================================================
    # TOKEN CHUNKING
    # ==================================================

    def _chunk(self, docs):

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n", "\n", ".", "•", "-", " "
            ]
        )

        chunks = splitter.split_documents(docs)

        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = f"{c.metadata['doc_id']}_chunk_{i}"

        log.info(f"✂️ chunks gerados: {len(chunks)}")
        return chunks

    def _normalize(self, text):

        # Normaliza unicode (remove ruído invisível)
        text = unicodedata.normalize("NFKD", text)
    
        # Remove caracteres estranhos
        text = text.replace("\x0c", " ")
    
        # Corrige hifenização quebrada de TXT
        text = re.sub(r"-\s*\n\s*", "", text)
    
        # Remove múltiplos espaços
        text = re.sub(r"\s+", " ", text)
    
        return text.strip().lower()


    # ==================================================
    # HASH DEDUP
    # ==================================================

    def _hash_dedup(self, docs):

        seen = set()
        unique = []
    
        for d in docs:
    
            clean = self._normalize(d.page_content)
    
            h = hashlib.sha256(clean.encode()).hexdigest()
    
            if h not in seen:
                seen.add(h)
                unique.append(d)
    
        log.info(f"🧹 hash removidos: {len(docs)-len(unique)}")
        return unique


    # ==================================================
    # SEMANTIC DEDUP
    # ==================================================

    def _semantic_dedup(self, docs, threshold=0.95):

        texts = [d.page_content for d in docs]
        embs = np.array(self.embeddings.embed_documents(texts))

        sims = cosine_similarity(embs)

        keep = []
        removed = set()

        for i in range(len(docs)):
            if i in removed:
                continue

            keep.append(docs[i])

            for j in range(i + 1, len(docs)):
                if sims[i][j] > threshold:
                    removed.add(j)

        log.info(f"🧹 semantic removidos: {len(docs)-len(keep)}")
        return keep
