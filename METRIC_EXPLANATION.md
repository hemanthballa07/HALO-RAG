# Verification and answer metrics

## Evidence-grounding metrics

- **Factual precision** is the fraction of generated claims that the verifier marks as
  entailed by the retrieved evidence.
- **Factual recall** is the fraction of ground-truth claims entailed by that evidence.
- **Hallucination rate** is the fraction of generated claims that are not entailed. An
  abstention has no asserted claims and is assigned a hallucination rate of zero.
- **Coverage** is the fraction of normalized ground-truth answer tokens found in the
  evidence used for generation.

## Answer-quality metrics

- **Exact Match** compares normalized generated and reference answers.
- **Answer F1** uses token-count overlap rather than unique-token sets.
- **BLEU-4** and **ROUGE-L** capture longer-form overlap.

## Composite and behavior metrics

- **Verified F1** is `answer F1 × factual precision`.
- **FEVER-style score** is the harmonic mean of supported-claim accuracy and evidence
  recall used by this implementation.
- **Abstention rate** records whether the generated response contains a configured
  insufficient-evidence phrase.

Factual precision and hallucination rate are production-observable because they need only
the answer and retrieved evidence. Exact Match, answer F1, factual recall, coverage, and
Verified F1 require references and are evaluation-only.
