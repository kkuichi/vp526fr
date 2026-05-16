# Hodnotenie jazykových modelov na slovenskom benchmarku SKLEP

Bakalárska práca — experimentálne porovnanie klasifikačného modelu Slovak BERT (fine-tuning) s generatívnymi modelmi (zero-shot) na úlohách benchmarku SKLEP.

## Popis projektu

Cieľom práce bolo otestovať a porovnať výkonnosť 5 jazykových modelov na 9 úlohách slovenského benchmarku SKLEP (Slovak Language Evaluation Platform):

**Testované modely:**
- Slovak BERT (gerulata/slovakbert) — fine-tuning na každej úlohe
- Slovak GPT-J-1.4B (Milos/slovak-gpt-j-1.4B) — zero-shot
- Mistral Small (mistral-small-latest) — zero-shot cez API
- Mistral Large (mistral-large-latest) — zero-shot cez API
- Open Mistral Nemo (open-mistral-nemo) — zero-shot cez API

**Testované úlohy SKLEP:**
sentiment-analysis, hate-speech, nli, rte, sts, ner-uner, ner-wikigoldsk, pos, question-answering

## Štruktúra projektu

```
sklep/
├── eval/scripts/              # Skripty na doladenie Slovak BERT (adaptované z Hugging Face)
│   ├── run_classification.py  # Klasifikačné úlohy (sentiment, hate-speech)
│   ├── run_glue.py            # GLUE-type úlohy (NLI, RTE, STS)
│   ├── run_ner.py             # Token-level úlohy (NER, POS)
│   ├── run_qa.py              # Question Answering
│   ├── trainer_qa.py          # Pomocná trieda pre QA
│   └── utils_qa.py            # Pomocné funkcie pre QA
├── generative/                # Skripty na zero-shot testovanie generatívnych modelov
│   ├── test_generative.py     # Slovak GPT-J — klasifikačné úlohy
│   ├── test_gptj_extra.py     # Slovak GPT-J — STS, NER, POS, QA
│   ├── test_mistral.py        # Mistral Small — 5 základných úloh
│   ├── test_mistral_large.py  # Mistral Large — 5 základných úloh
│   ├── test_nemo.py           # Mistral Nemo — 5 základných úloh
│   ├── test_mistral_extra.py  # Mistral Small — NER, NER-wiki, POS, QA
│   ├── test_mistral_pos_qa.py # Mistral Small — POS a QA
│   └── test_large_nemo_extra.py # Mistral Large a Nemo — NER, POS, QA
├── results/                   # Výsledky doladenia Slovak BERT
├── results_generative/        # Výsledky zero-shot testovania (JSON)
├── images/                    # Grafy priebehu trénovania a porovnania modelov
├── run_sklep_tests.py         # Hlavný skript na spustenie doladenia Slovak BERT
└── .gitignore
```

## Spustenie
??????

### Požiadavky

- Python 3.10
- Závislosti: `pip install transformers datasets evaluate torch matplotlib scipy mistralai`

### Doladenie Slovak BERT

```bash
# Spustenie všetkých úloh
python run_sklep_tests.py

# Spustenie jednej úlohy
python run_sklep_tests.py sentiment-analysis
```

Hyperparametre: learning rate = 2e-5, batch size = 8, epochy = 3, max seq length = 128 (384 pre QA).

### Zero-shot testovanie generatívnych modelov

```bash
# Slovak GPT-J (lokálne, bez API)
python generative/test_generative.py
python generative/test_gptj_extra.py

# Mistral modely (vyžaduje API kľúč — nastaviť v skripte)
python generative/test_mistral.py
python generative/test_mistral_large.py
python generative/test_nemo.py
python generative/test_mistral_extra.py
python generative/test_mistral_pos_qa.py
python generative/test_large_nemo_extra.py
```


## Dáta

Benchmark SKLEP: [huggingface.co/datasets/slovak-nlp/sklep](https://huggingface.co/datasets/slovak-nlp/sklep)
