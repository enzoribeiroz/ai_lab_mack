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
# Arquivo : llama/perplexity.py
# Síntese : Indicador de confiança da resposta. Calcula a log-probabilidade de
#           cada token gerado, a média dessas log-probabilidades e a
#           perplexidade PPL = exp(-média), usada como sinal auxiliar de
#           incerteza do modelo.
#
# Histórico de alterações:
#   2026-02-26 - Guilherme Martins Silva - Versão inicial (get_logprobs com
#                novo forward pass sobre a sequência completa).
#   2026-11-15 - Enzo Ribeiro - Método from_generation_logits(), que reaproveita
#                os logits brutos devolvidos por generate(output_logits=True)
#                e evita um segundo forward pass (economia de VRAM); cabeçalho
#                padronizado.
# =============================================================================
import torch
import torch.nn.functional as F

class CalculatePerplexity:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _summarize(self, generated_logprobs, generated_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)

        pairs = list(zip(tokens, generated_logprobs.tolist()))

        # métricas úteis
        mean_logprob = generated_logprobs.mean().item()
        perplexity = torch.exp(-generated_logprobs.mean()).item()

        return pairs, mean_logprob, perplexity

    def from_generation_logits(self, logits, generated_ids):
        """
        logits        -> outputs.logits de generate(output_logits=True): tupla com um
                         tensor [batch, vocab] por passo de geração (antes de
                         temperatura/top-k/top-p, ou seja, a distribuição real do modelo)
        generated_ids -> ids gerados (sem o prompt)
        """
        step_logits = torch.stack([l[0] for l in logits]).float()       # [passos, vocab]
        log_probs = F.log_softmax(step_logits, dim=-1)
        ids = generated_ids[: step_logits.shape[0]].to(log_probs.device)
        generated_logprobs = log_probs.gather(1, ids.unsqueeze(-1)).squeeze(-1)
        return self._summarize(generated_logprobs.cpu(), ids.cpu())

    def get_logprobs(self, sequences, input_len):
        """
        sequences -> output.sequences (tensor)
        input_len -> tamanho do prompt
        """

        with torch.no_grad():
            logits = self.model(sequences).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = sequences[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)

        token_logprobs = log_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # pegar só parte gerada
        generated_logprobs = token_logprobs[0][input_len-1:]
        generated_ids = sequences[0][input_len:]

        return self._summarize(generated_logprobs, generated_ids)
