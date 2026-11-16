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
# Arquivo : llama/answer_dataset.py
# Síntese : Conjunto de avaliação das RESPOSTAS geradas pelo agente (fim a fim).
#           Cobre o resultado esperado declarado na N1 ("responder corretamente
#           a perguntas objetivas sobre os 5 componentes e 20 princípios"):
#             - 1 pergunta sobre os 5 componentes;
#             - 20 perguntas, uma por princípio;
#             - 5 perguntas sobre os princípios de cada componente;
#             - 5 perguntas FORA do escopo do COSO ERM (o agente deve se abster).
#           Cada pergunta tem grupos de conceitos esperados; um grupo é
#           atendido se QUALQUER uma de suas variantes (PT ou EN, sem acentos)
#           aparecer na resposta. A nota é a fração de grupos atendidos.
#
# Histórico de alterações:
#   2026-11-15 - Enzo Ribeiro - Versão inicial (31 perguntas + função de nota).
#   2026-11-16 - Enzo Ribeiro - Amplia os sinais de abstenção ("não encontrei").
# =============================================================================
import unicodedata

# Títulos oficiais dos 20 princípios (extraídos na EDA da N1, Tabela 1)
PRINCIPLES = {
    1: "Exercises Board Risk Oversight",
    2: "Establishes Operating Structures",
    3: "Defines Desired Culture",
    4: "Demonstrates Commitment to Core Values",
    5: "Attracts, Develops, and Retains Capable Individuals",
    6: "Analyzes Business Context",
    7: "Defines Risk Appetite",
    8: "Evaluates Alternative Strategies",
    9: "Formulates Business Objectives",
    10: "Identifies Risk",
    11: "Assesses Severity of Risk",
    12: "Prioritizes Risks",
    13: "Implements Risk Responses",
    14: "Develops Portfolio View",
    15: "Assesses Substantial Change",
    16: "Reviews Risk and Performance",
    17: "Pursues Improvement in Enterprise Risk Management",
    18: "Leverages Information and Technology",
    19: "Communicates Risk Information",
    20: "Reports on Risk, Culture, and Performance",
}

# Conceitos-chave de cada princípio (variantes em inglês e português, sem acento)
PRINCIPLE_TERMS = {
    1: [["board", "conselho"], ["oversight", "supervis"]],
    2: [["structure", "estrutura"], ["operat", "operac"]],
    3: [["culture", "cultura"], ["desired", "desejad"]],
    4: [["core value", "valores"], ["commitment", "compromet", "compromisso"]],
    5: [["attract", "atrai", "atrair"], ["retain", "retem", "retenc", "reter"]],
    6: [["context", "contexto"]],
    7: [["appetite", "apetite"]],
    8: [["alternative", "alternativ"], ["strateg", "estrateg"]],
    9: [["formulat", "formula"], ["objective", "objetivo"]],
    10: [["identif"], ["risk", "risco"]],
    11: [["severity", "severidade", "gravidade"]],
    12: [["prioritiz", "prioriz"]],
    13: [["response", "resposta"]],
    14: [["portfolio", "carteira"]],
    15: [["substantial", "substanc", "significativ"], ["change", "mudanc", "alterac"]],
    16: [["review", "revis", "analisa"], ["performance", "desempenho"]],
    17: [["improvement", "melhori", "aprimor"]],
    18: [["technology", "tecnolog"], ["information", "informac"]],
    19: [["communicat", "comunica"]],
    20: [["report", "relat", "reporta"]],
}

COMPONENTS = {
    "Governance and Culture": ([1, 2, 3, 4, 5], [["governance", "governanc"], ["culture", "cultura"]]),
    "Strategy and Objective-Setting": ([6, 7, 8, 9], [["strateg", "estrateg"], ["objective", "objetivo"]]),
    "Performance": ([10, 11, 12, 13, 14], [["performance", "desempenho"]]),
    "Review and Revision": ([15, 16, 17], [["review", "revis"], ["revision", "revis", "correc"]]),
    "Information, Communication, and Reporting": ([18, 19, 20], [["information", "informac"], ["communication", "comunica"], ["reporting", "relat", "report"]]),
}

COMPONENT_PT = {
    "Governance and Culture": "Governança e Cultura (Governance and Culture)",
    "Strategy and Objective-Setting": "Estratégia e Definição de Objetivos (Strategy and Objective-Setting)",
    "Performance": "Desempenho (Performance)",
    "Review and Revision": "Revisão e Correção (Review and Revision)",
    "Information, Communication, and Reporting": "Informação, Comunicação e Relato (Information, Communication, and Reporting)",
}

# Sinais de abstenção. Observação: uma resposta que se abstém e DEPOIS responde mesmo assim
# (ex.: "não encontrei..., no entanto...") também recebe nota 1 aqui; esses casos são
# discutidos manualmente no relatório.
ABSTAIN_TERMS = [["nao encontr", "not found", "nao ha informac", "nao possuo informac", "nao contem",
                  "nao tenho informac", "nao foi possivel encontrar", "nao esta no contexto",
                  "fora do escopo"]]


def _component_of(n):
    return next(c for c, (ps, _) in COMPONENTS.items() if n in ps)


def build_answer_dataset():
    """Lista de dicionários: id, category, question, expected (grupos de conceitos)."""
    items = []

    # 1) Os cinco componentes
    items.append({
        "id": "C0",
        "category": "componentes",
        "question": "Quais são os cinco componentes do framework COSO ERM 2017?",
        "expected": [g for _, (_, groups) in COMPONENTS.items() for g in groups],
    })

    # 2) Um item por princípio: título + componente ao qual pertence
    for n, title in PRINCIPLES.items():
        comp = _component_of(n)
        items.append({
            "id": f"P{n}",
            "category": "principio",
            "question": f"O que diz o Princípio {n} do COSO ERM 2017 e a qual componente ele pertence?",
            "expected": PRINCIPLE_TERMS[n] + COMPONENTS[comp][1],
            "reference": f"Principle {n}: {title} ({comp})",
        })

    # 3) Princípios de cada componente
    for i, (comp, (ps, _)) in enumerate(COMPONENTS.items(), start=1):
        items.append({
            "id": f"K{i}",
            "category": "principios_do_componente",
            "question": f"Quais princípios fazem parte do componente {COMPONENT_PT[comp]} no COSO ERM 2017?",
            # basta citar o conceito central de cada princípio do componente
            "expected": [PRINCIPLE_TERMS[p][0] for p in ps],
            "reference": "; ".join(f"{p}. {PRINCIPLES[p]}" for p in ps),
        })

    # 4) Fora do escopo: o comportamento correto é se abster
    out_of_scope = [
        "Qual é a taxa Selic definida na última reunião do Copom?",
        "Quem venceu a Copa do Mundo de futebol de 2022?",
        "Quais são as sanções administrativas previstas no artigo 52 da LGPD?",
        "Quais controles compõem o Anexo A da ISO/IEC 27001:2022?",
        "Qual é a receita tradicional do pão de queijo mineiro?",
    ]
    for i, q in enumerate(out_of_scope, start=1):
        items.append({
            "id": f"F{i}",
            "category": "fora_do_escopo",
            "question": q,
            "expected": ABSTAIN_TERMS,
        })

    return items


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def score_answer(answer: str, expected_groups) -> float:
    """Fração dos grupos de conceitos esperados presentes na resposta (0 a 1)."""
    norm = _normalize(answer)
    hits = sum(any(_normalize(v) in norm for v in group) for group in expected_groups)
    return hits / len(expected_groups)
