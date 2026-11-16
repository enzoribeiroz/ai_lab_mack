# Agente Especialista em GRC

Assistente conversacional que responde perguntas sobre o framework **COSO ERM 2017** (*Enterprise Risk Management – Integrating with Strategy and Performance*). Ele combina **RAG** (busca MMR + re-ranking com cross-encoder) com um **LLM executado localmente** (Llama 3.2 3B Instruct em 4 bits), cita os trechos usados e estima a confiança de cada resposta pela **perplexidade**.

Projeto da disciplina **Inteligência Artificial** (7ºK, noite, Sistemas de Informação), **Universidade Presbiteriana Mackenzie**, Faculdade de Computação e Informática. Professor: Dr. Leandro Zerbinatti. Opção do projeto: *API ChatGPT* (LLM generativo), com um modelo aberto local no lugar da API comercial.

| Integrante | RA | E-mail |
|---|---|---|
| Guilherme Martins Silva | 10417140 | 10417140@mackenzista.com.br |
| Enzo Ribeiro | 10418262 | 10418262@mackenzista.com.br |
| Pedro Nogueira Ribeiro | 10324924 | 10324924@mackenzista.com.br |

- **Relatório final (N2):** [`Relatorio_N2_Agente_Especialista_GRC.pdf`](Relatorio_N2_Agente_Especialista_GRC.pdf) (versão editável em `.docx`)
- **Relatório N1:** [`Relatorio_N1_Agente_Especialista_GRC.pdf`](Relatorio_N1_Agente_Especialista_GRC.pdf)
- **Vídeo de apresentação:** https://youtu.be/yG522ogJRS0

## Dataset

| Arquivo | Conteúdo |
|---|---|
| `coso_erm_2017.txt` | Texto do COSO ERM 2017 extraído do PDF oficial (eBook ISBN 978-1-94549-886-2), em inglês. Cerca de 367 mil caracteres, 2.492 sentenças, 5 componentes e 20 princípios. É a base da análise exploratória. |
| `llama/RAG/docs_uploader/documents_txt/` | O mesmo texto, indexado pelo RAG. |
| `llama/RAG/db/` | Índice vetorial (ChromaDB) com 293 chunks de até 400 tokens, embeddings BAAI/bge-m3. |
| `llama/eval_dataset.py` | 16 perguntas em português, cada uma com sua tradução para o inglês e o trecho de referência (gabarito). Usado para avaliar a recuperação. |
| `llama/answer_dataset.py` | 31 perguntas para avaliar as respostas: os 5 componentes, os 20 princípios, os princípios de cada componente e 5 perguntas fora do escopo. |

O documento não contém dados pessoais. O texto do COSO é protegido por direitos autorais e está aqui **apenas para reprodução acadêmica** dos experimentos (detalhes na seção 4 do relatório).

## Estrutura

```
├── main.ipynb                     # demonstração do agente (pergunta, acompanhamento, fora do escopo)
├── EDA_COSO_ERM.ipynb             # análise exploratória do corpus (N1)
├── requirements.txt
└── llama/
    ├── call_llama.py              # orquestrador: memória + RAG + Llama + perplexidade
    ├── perplexity.py              # log-probabilidade média e perplexidade
    ├── EvaluationRAG.py           # Hit Rate, MRR, Precision@1
    ├── baselines.py               # aleatório, BM25, denso, MMR, re-ranking
    ├── eval_dataset.py            # 16 perguntas de recuperação + gabarito
    ├── answer_dataset.py          # 31 perguntas de avaliação das respostas
    ├── evaluateRAG.ipynb          # avaliação original do recuperador
    ├── experimentos_N2.ipynb      # baselines, tamanho de chunk, avaliação fim a fim
    ├── resultados/                # CSVs e figuras gerados pelos notebooks
    ├── RAG/
    │   ├── BuildVectorStore.py    # ingestão: chunking, deduplicação, Chroma
    │   ├── rank_context.py        # re-ranker (cross-encoder)
    │   ├── text_cleaning.py       # limpeza do texto (ligature "fi"), vinda da EDA
    │   ├── generate_db.ipynb      # (re)geração do índice
    │   └── db/                    # índice persistido
    ├── memory_support/            # memória conversacional persistente
    ├── memory_support_db/
    └── logs_llama/                # log das interações
```

## Como executar

Requisitos: GPU NVIDIA com pelo menos 6 GB de VRAM e Python 3.11 ou superior (testado com 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

O repositório oficial `meta-llama/Llama-3.2-3B-Instruct` exige que você aceite a licença no HuggingFace e faça `hf auth login`. A alternativa é usar o espelho público com os mesmos pesos:

```bash
export LLAMA_MODEL_ID=unsloth/Llama-3.2-3B-Instruct
```

| O que fazer | Como |
|---|---|
| Conversar com o agente | `main.ipynb`, a partir da raiz |
| Reproduzir a avaliação do recuperador | `llama/evaluateRAG.ipynb`, a partir de `llama/` |
| Reproduzir os experimentos da N2 | `llama/experimentos_N2.ipynb`, a partir de `llama/` (cerca de 20 min numa RTX 4050) |
| Regerar o índice vetorial | apagar `llama/RAG/db/*` e executar `llama/RAG/generate_db.ipynb` |

Em GPUs com menos de 8 GB, o `main.ipynb` passa sozinho os embeddings e o re-ranker para a CPU.

## Resultados principais

Gerados por `llama/experimentos_N2.ipynb` (CSVs e figuras em `llama/resultados/`). Análise completa na seção 7 do relatório.

**Recuperação** (16 perguntas em português, 6 trechos por pergunta). Colunas "tolerante" com threshold 0,77 e "estrito" com threshold 0,85:

| Recuperador | Hit Rate (tolerante) | MRR (tolerante) | Hit Rate (estrito) | MRR (estrito) |
|---|---|---|---|---|
| Aleatório | 0,38 | 0,16 | 0,12 | 0,02 |
| BM25 (pergunta em PT) | 0,19 | 0,14 | 0,12 | 0,04 |
| BM25 (pergunta traduzida para EN) | 0,81 | 0,58 | 0,31 | 0,20 |
| Denso (bge-m3) | 0,81 | 0,63 | 0,25 | 0,18 |
| Denso + re-ranking | 0,94 | 0,73 | 0,56 | 0,30 |
| **MMR + re-ranking (agente)** | **0,81** | **0,63** | **0,50** | **0,32** |

- A busca semântica supera a lexical. Com as perguntas em português, o BM25 fica abaixo do acaso, porque o corpus está em inglês.
- O re-ranking com cross-encoder é a etapa que mais agrega: dobra a taxa de acerto do trecho exato.
- O MMR não trouxe ganho neste corpus. Denso + re-ranking foi melhor.
- Chunks de 400 tokens são o melhor compromisso entre 200 (mais precisos, menor cobertura) e 800 (maior cobertura, pior MRR). A limpeza do texto não melhorou a recuperação.

**Respostas do LLM** (31 perguntas, memória desligada):

- As respostas cobriram em média 55% dos conceitos esperados. O erro mais comum é confundir o número do princípio, porque o trecho "Principle N" só foi recuperado em 10 das 20 perguntas.
- Fora do escopo: o agente se absteve corretamente em parte das 5 perguntas e inventou respostas nas demais.
- A perplexidade ficou entre 1,08 e 1,36 e **não se correlacionou** com a qualidade da resposta (Spearman ≈ 0). Ela não serve, sozinha, como indicador de confiança.
- Tempo médio de resposta: cerca de 5 s numa RTX 4050 de 6 GB.

## Cabeçalho e histórico

Todo arquivo-fonte (`.py` e `.ipynb`) começa com a identificação dos integrantes, uma síntese do conteúdo e o histórico de alterações (data, autor, descrição), como pede o enunciado do projeto.
