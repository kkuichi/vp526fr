import json, os, random, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Milos/slovak-gpt-j-1.4B"
MAX_SAMPLES = 50
RESULTS_DIR = "results_generative"
random.seed(42)

print("Zavadzujem model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()
print("Model hotovy!\n")

def generate(prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


print("Zadaca: sts")
ds_sts = load_dataset("slovak-nlp/sklep", "sts", split="test")
indices = list(range(len(ds_sts)))
random.shuffle(indices)
indices = indices[:MAX_SAMPLES]

preds = []
trues = []
start = time.time()
for i, idx in enumerate(indices):
    ex = ds_sts[idx]
    prompt = f'Ohodnot podobnost dvoch viet cislom od 0 do 5.\n\nVeta 1: "{ex["sentence1"][:200]}"\nVeta 2: "{ex["sentence2"][:200]}"\n\nPodobnost:'
    response = generate(prompt, max_new_tokens=5)
    try:
        pred = float(response.strip().replace(",", ".").split()[0])
        if pred > 5: pred = 5
        if pred < 0: pred = 0
    except:
        pred = 2.5
    preds.append(pred)
    trues.append(ex["similarity_score"])
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}]")

from scipy.stats import pearsonr
pearson_val, _ = pearsonr(trues, preds)
print(f"\n  REZULTAT Pearson: {pearson_val:.4f}")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")
with open(os.path.join(RESULTS_DIR, "sts_gptj.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "sts", "model": MODEL_NAME, "pearson": round(pearson_val, 4)}, f, indent=2, ensure_ascii=False)

print("Zadaca: ner-uner")

ds_ner = load_dataset("slovak-nlp/sklep", "ner-uner", split="test")
label_names = ds_ner.features["ner_tags"].feature.names
indices = list(range(len(ds_ner)))
random.shuffle(indices)
indices = indices[:MAX_SAMPLES]
correct = 0
total = 0
start = time.time()
for i, idx in enumerate(indices):
    ex = ds_ner[idx]
    tokens = ex["tokens"][:30]
    true_tags = [label_names[t] for t in ex["ner_tags"]][:30]
    prompt = f'Pre kazdy token urc NER tag (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).\n\nTokeny: {" ".join(tokens)}\n\nNER tagy:'
    response = generate(prompt, max_new_tokens=60)
    pred = response.strip().split()
    min_len = min(len(true_tags), len(pred))
    for j in range(min_len):
        if pred[j].upper() == true_tags[j].upper():
            correct += 1
        total += 1
    total += max(0, len(true_tags) - len(pred))
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}] {correct/total*100:.1f}%")
ner_acc = correct/total*100 if total > 0 else 0
print(f"\n  REZULTAT: {ner_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")
with open(os.path.join(RESULTS_DIR, "ner-uner_gptj.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "ner-uner", "model": MODEL_NAME, "token_accuracy": round(ner_acc, 2)}, f, indent=2, ensure_ascii=False)

print("Zadaca: ner-wikigoldsk")

ds_wiki = load_dataset("slovak-nlp/sklep", "ner-wikigoldsk", split="test")
label_names_w = ds_wiki.features["ner_tags"].feature.names
indices = list(range(len(ds_wiki)))
random.shuffle(indices)
indices = indices[:MAX_SAMPLES]
correct = 0
total = 0
start = time.time()
for i, idx in enumerate(indices):
    ex = ds_wiki[idx]
    tokens = ex["tokens"][:30]
    true_tags = [label_names_w[t] for t in ex["ner_tags"]][:30]
    prompt = f'Pre kazdy token urc NER tag (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).\n\nTokeny: {" ".join(tokens)}\n\nNER tagy:'
    response = generate(prompt, max_new_tokens=60)
    pred = response.strip().split()
    min_len = min(len(true_tags), len(pred))
    for j in range(min_len):
        if pred[j].upper() == true_tags[j].upper():
            correct += 1
        total += 1
    total += max(0, len(true_tags) - len(pred))
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}] {correct/total*100:.1f}%")
wiki_acc = correct/total*100 if total > 0 else 0
print(f"\n  REZULTAT: {wiki_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")
with open(os.path.join(RESULTS_DIR, "ner-wikigoldsk_gptj.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "ner-wikigoldsk", "model": MODEL_NAME, "token_accuracy": round(wiki_acc, 2)}, f, indent=2, ensure_ascii=False)


print("Zadaca: pos")

ds_pos = load_dataset("slovak-nlp/sklep", "pos", split="test")
indices = list(range(len(ds_pos)))
random.shuffle(indices)
indices = indices[:MAX_SAMPLES]
correct = 0
total = 0
start = time.time()
for i, idx in enumerate(indices):
    ex = ds_pos[idx]
    tokens = ex["tokens"][:30]
    true_tags = ex["pos_tags"][:30]
    prompt = f'Pre kazdy token urc POS tag (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CCONJ, SCONJ, PART, AUX, PUNCT, PROPN, X, SYM, INTJ).\n\nTokeny: {" ".join(tokens)}\n\nPOS tagy:'
    response = generate(prompt, max_new_tokens=60)
    pred = response.strip().split()
    min_len = min(len(true_tags), len(pred))
    for j in range(min_len):
        if pred[j].upper() == true_tags[j].upper():
            correct += 1
        total += 1
    total += max(0, len(true_tags) - len(pred))
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}] {correct/total*100:.1f}%")
pos_acc = correct/total*100 if total > 0 else 0
print(f"\n  REZULTAT: {pos_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")
with open(os.path.join(RESULTS_DIR, "pos_gptj.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "pos", "model": MODEL_NAME, "token_accuracy": round(pos_acc, 2)}, f, indent=2, ensure_ascii=False)


print("Zadaca: question-answering")

ds_qa = load_dataset("slovak-nlp/sklep", "question-answering", split="test")
indices = list(range(len(ds_qa)))
random.shuffle(indices)
indices = indices[:MAX_SAMPLES]
correct = 0
total = 0
start = time.time()
for i, idx in enumerate(indices):
    ex = ds_qa[idx]
    prompt = f'Na zaklade kontextu odpoved na otazku.\n\nKontext: {ex["context"][:300]}\n\nOtazka: {ex["question"]}\n\nOdpoved:'
    response = generate(prompt, max_new_tokens=30)
    answers = ex["answers"]
    is_correct = False
    if answers and "text" in answers and len(answers["text"]) > 0:
        for ans in answers["text"]:
            if ans.lower() in response.lower() or response.lower() in ans.lower():
                is_correct = True
                break
    if is_correct:
        correct += 1
    total += 1
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}] {correct/total*100:.1f}%")
qa_acc = correct/total*100
print(f"\n  REZULT: {qa_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")
with open(os.path.join(RESULTS_DIR, "qa_gptj.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "question-answering", "model": MODEL_NAME, "accuracy": round(qa_acc, 2)}, f, indent=2, ensure_ascii=False)


print("Suhrn Slovak GPT-J-1.4B:")
print(f"  sts (Pearson)          {pearson_val:.4f}")
print(f"  ner-uner               {ner_acc:.2f}%")
print(f"  ner-wikigoldsk         {wiki_acc:.2f}%")
print(f"  pos                    {pos_acc:.2f}%")
print(f"  question-answering     {qa_acc:.2f}%")

