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
# Arquivo : llama/call_llama.py
# Síntese : Orquestrador do agente. Carrega re-ranker, memória conversacional,
#           banco vetorial do RAG e o Llama 3.2 3B Instruct quantizado em 4 bits;
#           para cada pergunta recupera memórias e trechos do COSO ERM (MMR +
#           re-ranking), monta o prompt, gera a resposta, calcula a perplexidade
#           e registra a interação em log e na memória.
#
# Histórico de alterações:
#   2026-02-26 - Guilherme Martins Silva - Versão inicial do orquestrador.
#   2026-11-15 - Enzo Ribeiro - Prompt montado com o chat template oficial do
#                Llama 3.2 (o texto anterior usava <|end_header_id|> no lugar de
#                <|eot_id|> e duplicava o <|begin_of_text|>); perplexidade
#                calculada a partir dos logits da própria geração; métricas da
#                última resposta em self.last_metrics; opção de desligar a memória
#                (avaliação sem contaminação); seed; caminhos independentes do
#                cwd; seleção automática de dispositivo; cabeçalho padronizado.
# =============================================================================
import os
import json
import torch
from tqdm import tqdm
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings

from llama.memory_support.memory_support import Memory_support
from llama.RAG.BuildVectorStore import PDFVectorStore
from llama.RAG.rank_context import RankContext
from llama.perplexity import CalculatePerplexity


SYSTEM_PROMPT = """Você é um especialista em GRC.

Construa uma resposta completa e explicativa usando todas as informações relevantes disponíveis.

Responda utilizando as informações do <CONTEXT_RAG> como base factual.
Use também o <CONTEXT_MEMORY> se complementar a resposta.

Não invente fatos que não estejam no contexto.

Se múltiplos documentos se complementarem, combine-os.

Se não houver informação suficiente, responda exatamente:
Informação não encontrada no contexto.

<CONTEXT_MEMORY>
{memory_block}
</CONTEXT_MEMORY>

<CONTEXT_RAG>
{rag_block}
</CONTEXT_RAG>"""


class CallLlama:
    def __init__(self,
                 model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device_map: str = None,
                 log_file: str = "logs_llama/llama_logs.jsonl",
                 use_memory: bool = True,
                 aux_device: str = None,
                 verbose: bool = True):

        print("🚀 Inicializando call_llama...")

        # Configurações de caminhos base (tudo relativo à pasta llama/, independente do cwd)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_id = model_id
        self.device_map = device_map or ("cuda" if torch.cuda.is_available() else "cpu")
        # Dispositivo dos modelos auxiliares (embeddings, re-ranker); "cpu" libera VRAM para o LLM
        self.aux_device = aux_device or self.device_map
        self.use_memory = use_memory
        self.verbose = verbose
        self.log_file = os.path.join(self.base_dir, log_file)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.memory_path = os.path.join(self.base_dir, "memory_support_db")
        os.makedirs(self.memory_path, exist_ok=True)
        self.last_metrics = {}

        # Definição das etapas de carregamento
        steps = [
            {"desc": "🔍 Carregando sistema de Re Rank", "func": self._init_rerank},
            {"desc": "🧠 Iniciando sistema de memória", "func": self._init_memory},
            {"desc": "📚 Iniciando sistema de RAG", "func": self._init_rag},
            {"desc": "⚙️ Carregando modelo Llama local", "func": self.instanciate_llama},
            {"desc": "📊 Inicializando Answer Evaluation", "func": self._init_perplexity}
        ]

        # Barra de progresso principal
        with tqdm(total=len(steps), desc="Progresso Geral", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for step in steps:
                pbar.set_description(step["desc"])
                step["func"]()  # Executa a função de inicialização
                pbar.update(1)

        print("\n✅ call_llama inicializado com sucesso!")

    # --- Métodos Auxiliares para organizar o código ---

    def _init_rerank(self):
        self.rank_context = RankContext(model="BAAI/bge-reranker-v2-m3",
                                        device=self.aux_device, verbose=self.verbose)

    def _init_memory(self):
        if not self.use_memory:
            self.memory_support = None
            return
        self.memory_support = Memory_support(
            db_path=self.memory_path,
            embedding_model=HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-small",
                model_kwargs={'device': self.aux_device}
            ),
            threshold_distance=0.70,
            max_memories=300
        )

    def _init_rag(self):
        self.rag_support = PDFVectorStore(
            persist_dir=os.path.join(self.base_dir, "RAG", "db"),
            device=self.aux_device
        )
        self.db_rag = self.rag_support._load_db()

    def _init_perplexity(self):
        # Garante que o modelo e tokenizer já existem antes de chamar
        self.calculate_perplexity = CalculatePerplexity(model=self.model, tokenizer=self.tokenizer)

    # -------------------------------------------------------------------------
    #   LOAD LLAMA
    # -------------------------------------------------------------------------
    def instanciate_llama(self):
        # Quantização NF4 em 4 bits (bitsandbytes): ~2,3 GB de VRAM para o modelo de 3B
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map=self.device_map,
            dtype=torch.bfloat16
        )
        self.model.eval()

    # -------------------------------------------------------------------------
    #   LOG INTERAÇÕES
    # -------------------------------------------------------------------------
    def _log_interaction(self, user, response):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "response": response,
            "mean_logprob": self.last_metrics.get("mean_logprob"),
            "perplexity": self.last_metrics.get("perplexity"),
            "sources": self.last_metrics.get("sources"),
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _extract_info_chunks(self, user, k):

        # Estágio 1: MMR busca 50 candidatos por similaridade e devolve k diversos
        relevant_chunks = self.db_rag.max_marginal_relevance_search(
            user,
            k=k,        # Quantidade para o Re-ranker analisar
            fetch_k=50,  # Quantidade que o Chroma analisa internamente
            lambda_mult=0.50
        )

        # Estágio 2: o cross-encoder reordena e mantém os top_n (6)
        reranked_docs = self.rank_context.call_rank(query=user, documents=relevant_chunks)

        return reranked_docs

    # -------------------------------------------------------------------------
    #   MONTA PROMPT E GERA RESPOSTA
    # -------------------------------------------------------------------------
    def build_prompt(self, user: str, rag_docs=None):

        memory_block = self.memory_support.mount_prompt_memory(user, k=3) if self.memory_support else ""
        if rag_docs is None:
            rag_docs = self._extract_info_chunks(user, k=20)
        self._last_docs = rag_docs

        if rag_docs:
            rag_block = "\n\n".join(
                f"[FONTE {d.id}] {d.page_content}"
                for d in rag_docs
            )
        else:
            rag_block = ""

        if self.verbose:
            print(f"RAG_BLOCK: {rag_block}")

        system = SYSTEM_PROMPT.format(
            memory_block=memory_block if memory_block.strip() else "[Nenhuma memória relevante encontrada]",
            rag_block=rag_block if rag_block.strip() else "[Nenhum documento relevante foi encontrado.]",
        )

        # Usa o chat template oficial do modelo: garante os tokens especiais corretos
        # (<|start_header_id|>, <|eot_id|>) e um único <|begin_of_text|>.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    # -------------------------------------------------------------------------
    #   INVOCAÇÃO PRINCIPAL
    # -------------------------------------------------------------------------
    def invoke(self, user: str, temp=0.3, max_tokens=512, top_p=0.7, top_k=40,
               seed=None, rag_docs=None, save_memory=True):

        if seed is not None:
            torch.manual_seed(seed)

        prompt_main = self.build_prompt(user, rag_docs=rag_docs)

        # =============================
        # GERAR RESPOSTA
        # =============================
        # O template já contém <|begin_of_text|>, então não adicionamos tokens especiais de novo
        inputs_main = self.tokenizer(prompt_main, return_tensors="pt",
                                     add_special_tokens=False).to(self.model.device)

        with torch.no_grad():
            outputs_main = self.model.generate(
                **inputs_main,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                return_dict_in_generate=True,
                output_logits=True,  # logits brutos de cada passo -> perplexidade sem 2º forward
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_len_main = inputs_main["input_ids"].shape[1]
        generated_ids_main = outputs_main.sequences[0][input_len_main:]
        answer = self.tokenizer.decode(generated_ids_main, skip_special_tokens=True).strip()

        # =============================
        # PERPLEXITY
        # =============================
        pairs, mean_lp, ppl = self.calculate_perplexity.from_generation_logits(
            logits=outputs_main.logits,
            generated_ids=generated_ids_main
        )

        self.last_metrics = {
            "mean_logprob": mean_lp,
            "perplexity": ppl,
            "n_tokens": len(pairs),
            "prompt_tokens": input_len_main,
            "sources": [d.metadata.get("chunk_id") for d in (self._last_docs or [])],
        }

        # ---- MEMÓRIA ----
        if self.memory_support is not None and save_memory:
            try:
                self.memory_support.append_on_memory_database(user, answer)
            except Exception as e:
                print(f"⚠️ Erro ao salvar memória: {e}")

        # ---- LOG ----
        self._log_interaction(user, answer)

        if self.verbose:
            print("\nPairs logprob x tokens\n", pairs)
        print(f"\nMean logprob: {mean_lp:.4f}")
        print(f"Perplexity: {ppl:.4f}")

        return answer
