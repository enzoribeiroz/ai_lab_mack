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
# Arquivo : llama/RAG/text_cleaning.py
# Síntese : Limpeza do texto extraído do PDF do COSO ERM, levada da análise
#           exploratória da N1 (EDA_COSO_ERM.ipynb) para o pipeline de RAG:
#           remove os marcadores "break" de paginação e corrige o ligature "fi"
#           perdido na extração (ex.: "Identi es" -> "Identifies") usando o
#           vocabulário do NLTK. Usada no experimento "corpus limpo x bruto".
#
# Histórico de alterações:
#   2026-11-15 - Enzo Ribeiro - Versão inicial, extraída do notebook de EDA
#                (mesma heurística descrita na seção 5.2 do relatório N1).
# =============================================================================
import re

import nltk
from nltk.corpus import words as nltk_words

_VOCAB = None


def _vocab():
    """Carrega (uma única vez) o vocabulário inglês do NLTK."""
    global _VOCAB
    if _VOCAB is None:
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words", quiet=True)
        _VOCAB = set(w.lower() for w in nltk_words.words())
    return _VOCAB


def known_word(word: str) -> bool:
    """Verifica se a palavra (ou uma flexão simples dela) existe no vocabulário inglês."""
    vocab = _vocab()
    w = word.lower()
    if w in vocab:
        return True
    if w.endswith("ies") and (w[:-3] + "y") in vocab:  # identifies -> identify
        return True
    for suffix in ("es", "ed", "s", "ing"):
        if w.endswith(suffix) and w[: -len(suffix)] in vocab:
            return True
    return False


def fix_broken_fi_ligature(text: str):
    """Funde pares 'Identi es' -> 'Identifies' quando a fusão com 'fi' é uma palavra
    válida e a simples junção sem 'fi' não é. Retorna (texto, nº de correções)."""
    pattern = re.compile(r"\b([A-Za-z]{2,})[ ]([a-z]{1,6})\b")
    fixes = 0

    def repl(match):
        nonlocal fixes
        left, right = match.group(1), match.group(2)
        candidate = f"{left}fi{right}"
        merged = f"{left}{right}"
        if known_word(candidate) and merged.lower() not in _vocab():
            fixes += 1
            return candidate
        return match.group(0)

    return pattern.sub(repl, text), fixes


def clean_text(text: str) -> str:
    """Pipeline de limpeza completo; devolve apenas o texto limpo."""
    text = text.replace("\r\n", "\n")
    no_breaks = re.sub(r"(?m)^\s*break\s*$\n?", "", text)
    fixed, _ = fix_broken_fi_ligature(no_breaks)
    normalized = re.sub(r"[ \t]+", " ", fixed)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized
