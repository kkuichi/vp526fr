import json, os, random, time
from datasets import load_dataset
from mistralai import Mistral

API_KEY = "xdsB8Vwt6tkVc8Lx6WajOsmIDIu7uDFV"
MODEL_NAME = "open-mistral-nemo"
MAX_SAMPLES = 100
RESULTS_DIR = "results_generative"
os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(42)

client = Mistral(api_key=API_KEY)
print(f"Model: {MODEL_NAME}\n")

def ask(prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Retry {attempt+1}... ({e})")
            time.sleep(10)
    return ""

def evaluate_task(task_name, dataset_config, prompt_fn, parse_fn, split="test"):
   
    print(f"{task_name}")
   
    ds = load_dataset("slovak-nlp/sklep", dataset_config, split=split)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:MAX_SAMPLES]
    correct = 0
    total = 0
    results = []
    start = time.time()
    for i, idx in enumerate(indices):
        example = ds[idx]
        prompt = prompt_fn(example)
        response = ask(prompt)
        predicted = parse_fn(response)
        true_label = example["label"]
        is_correct = (predicted == true_label)
        if is_correct:
            correct += 1
        total += 1
        results.append({"true": true_label, "pred": predicted, "response": response[:100], "correct": is_correct})
        if (i+1) % 10 == 0:
            acc = correct/total*100
            eta = (time.time()-start)/(i+1)*(len(indices)-i-1)/60
            print(f"  [{i+1}/{len(indices)}] Accuracy: {acc:.1f}% | ETA: {eta:.1f} min")
        time.sleep(1)
    acc = correct/total*100
    print(f"\n  REZULTAT: {acc:.2f}% ({correct}/{total})")
    print(f"  Cas: {(time.time()-start)/60:.1f} min\n")
    path = os.path.join(RESULTS_DIR, f"{task_name}_nemo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"task": task_name, "model": MODEL_NAME, "accuracy": acc, "correct": correct, "total": total, "predictions": results}, f, indent=2, ensure_ascii=False)
    return acc

def sent_prompt(ex):
    return f'Urc sentiment nasledujuceho textu. Odpoved je JEDNO slovo: pozitivny alebo negativny.\n\nText: "{ex["text"][:500]}"\n\nSentiment:'
def sent_parse(r):
    r = r.lower()
    if "pozit" in r: return 1
    if "negat" in r: return 0
    return -1
acc1 = evaluate_task("sentiment-analysis", "sentiment-analysis", sent_prompt, sent_parse)


def hate_prompt(ex):
    return f'Je nasledujuci text nenavistny (hate speech)? Odpoved je JEDNO slovo: ano alebo nie.\n\nText: "{ex["text"][:500]}"\n\nOdpoved:'
def hate_parse(r):
    r = r.lower()
    if "nie" in r: return 0
    if "ano" in r or "áno" in r: return 1
    return -1
acc2 = evaluate_task("hate-speech", "hate-speech", hate_prompt, hate_parse)

def nli_prompt(ex):
    return f'Urc vztah medzi vetami. Odpoved je JEDNO slovo: implikacia, neutralny alebo rozpor.\n\nVeta 1: "{ex["premise"][:300]}"\nVeta 2: "{ex["hypothesis"][:300]}"\n\nVztah:'
def nli_parse(r):
    r = r.lower()
    if "implik" in r or "vyplyv" in r: return 0
    if "neutr" in r: return 1
    if "rozp" in r or "kontrad" in r: return 2
    return -1
acc3 = evaluate_task("nli", "nli", nli_prompt, nli_parse)

def rte_prompt(ex):
    return f'Vyplyva druha veta z prvej? Odpoved je JEDNO slovo: ano alebo nie.\n\nVeta 1: "{ex["text1"][:300]}"\nVeta 2: "{ex["text2"][:300]}"\n\nOdpoved:'
def rte_parse(r):
    r = r.lower()
    if "nie" in r: return 1
    if "ano" in r or "áno" in r: return 0
    return -1
acc4 = evaluate_task("rte", "rte", rte_prompt, rte_parse)

import numpy as np
from scipy.stats import pearsonr
ds_sts = load_dataset("slovak-nlp/sklep", "sts", split="test")
indices_sts = list(range(len(ds_sts)))
random.shuffle(indices_sts)
indices_sts = indices_sts[:MAX_SAMPLES]


print(f"sts")

preds_list = []
trues_list = []
start = time.time()
for i, idx in enumerate(indices_sts):
    ex = ds_sts[idx]
    prompt = f'Ohodnot podobnost dvoch viet na skale 0 az 5 (0=uplne odlisne, 5=rovnake). Odpoved je JEDNO cislo.\n\nVeta 1: "{ex["sentence1"][:300]}"\nVeta 2: "{ex["sentence2"][:300]}"\n\nPodobnost:'
    response = ask(prompt)
    try:
        pred = float(response.strip().replace(",", ".").split()[0])
    except:
        pred = 2.5
    preds_list.append(pred)
    trues_list.append(ex["similarity_score"])
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}]")
    time.sleep(1)

pearson_val, _ = pearsonr(trues_list, preds_list)
print(f"\n  REZULTAT Pearson: {pearson_val:.4f}")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")

with open(os.path.join(RESULTS_DIR, "sts_nemo.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "sts", "model": MODEL_NAME, "pearson": round(pearson_val, 4)}, f, indent=2, ensure_ascii=False)

print(f"PIDSUIMOK: {MODEL_NAME}")

print(f"  sentiment-analysis    {acc1:.2f}%")
print(f"  hate-speech           {acc2:.2f}%")
print(f"  nli                   {acc3:.2f}%")
print(f"  rte                   {acc4:.2f}%")
print(f"  sts (Pearson)         {pearson_val:.4f}")


with open(os.path.join(RESULTS_DIR, "summary_nemo.json"), "w", encoding="utf-8") as f:
    json.dump({"model": MODEL_NAME, "results": {"sentiment-analysis": acc1, "hate-speech": acc2, "nli": acc3, "rte": acc4, "sts_pearson": round(pearson_val, 4)}}, f, indent=2, ensure_ascii=False)
print("Hotovo!")
