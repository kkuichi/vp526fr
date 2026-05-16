import json, os, random, time
from datasets import load_dataset
from mistralai import Mistral

API_KEY = "xdsB8Vwt6tkVc8Lx6WajOsmIDIu7uDFV"
MAX_SAMPLES = 50
RESULTS_DIR = "results_generative"
random.seed(42)
client = Mistral(api_key=API_KEY)

def ask(model_name, prompt, max_tokens=200, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.complete(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Retry {attempt+1}... ({e})")
            time.sleep(10)
    return ""

def test_ner(model_name, task, dataset_config, label_key="ner_tags"):
    
    print(f"{model_name} -> {task}")
    
    ds = load_dataset("slovak-nlp/sklep", dataset_config, split="test")
    if hasattr(ds.features[label_key], "feature") and hasattr(ds.features[label_key].feature, "names"):
        get_tags = lambda ex: [ds.features[label_key].feature.names[t] for t in ex[label_key]]
    else:
        get_tags = lambda ex: ex[label_key]
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:MAX_SAMPLES]
    correct = 0
    total = 0
    start = time.time()
    for i, idx in enumerate(indices):
        ex = ds[idx]
        tokens = ex["tokens"][:50]
        true_tags = get_tags(ex)[:50]
        prompt = f"""Pre kazdy token urc NER tag. Pouzi IOB format: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.
Odpoved: zoznam tagov oddelenych medzerami, PRESNE {len(tokens)} tagov.

Tokeny: {" ".join(tokens)}

NER tagy:"""
        response = ask(model_name, prompt, max_tokens=300)
        pred = response.strip().split()
        min_len = min(len(true_tags), len(pred))
        for j in range(min_len):
            if pred[j].upper() == true_tags[j].upper():
                correct += 1
            total += 1
        total += max(0, len(true_tags) - len(pred))
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{MAX_SAMPLES}] {correct/total*100:.1f}%")
        time.sleep(1)
    acc = correct/total*100 if total > 0 else 0
    print(f"  REZULTAT: {acc:.2f}%")
    suffix = model_name.replace("mistral-", "").replace("open-mistral-", "")
    with open(os.path.join(RESULTS_DIR, f"{task}_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"task": task, "model": model_name, "token_accuracy": round(acc, 2)}, f, indent=2, ensure_ascii=False)
    return acc

def test_pos(model_name):
    print(f"{model_name} -> pos")
    
    ds = load_dataset("slovak-nlp/sklep", "pos", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:MAX_SAMPLES]
    correct = 0
    total = 0
    start = time.time()
    for i, idx in enumerate(indices):
        ex = ds[idx]
        tokens = ex["tokens"][:50]
        true_tags = ex["pos_tags"][:50]
        prompt = f"""Pre kazdy token urc POS tag: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CCONJ, SCONJ, PART, INTJ, AUX, PUNCT, SYM, X, PROPN.
Odpoved: zoznam tagov oddelenych medzerami, PRESNE {len(tokens)} tagov.

Tokeny: {" ".join(tokens)}

POS tagy:"""
        response = ask(model_name, prompt, max_tokens=300)
        pred = response.strip().split()
        min_len = min(len(true_tags), len(pred))
        for j in range(min_len):
            if pred[j].upper() == true_tags[j].upper():
                correct += 1
            total += 1
        total += max(0, len(true_tags) - len(pred))
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{MAX_SAMPLES}] {correct/total*100:.1f}%")
        time.sleep(1)
    acc = correct/total*100 if total > 0 else 0
    print(f"  REZULTAT: {acc:.2f}%")
    suffix = model_name.replace("mistral-", "").replace("open-mistral-", "")
    with open(os.path.join(RESULTS_DIR, f"pos_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"task": "pos", "model": model_name, "token_accuracy": round(acc, 2)}, f, indent=2, ensure_ascii=False)
    return acc

def test_qa(model_name):
    print(f"{model_name} -> question-answering")
    
    ds = load_dataset("slovak-nlp/sklep", "question-answering", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:MAX_SAMPLES]
    correct = 0
    total = 0
    for i, idx in enumerate(indices):
        ex = ds[idx]
        prompt = f"""Na zaklade kontextu odpoved na otazku. Ak odpoved nie je v texte, napis "ziadna odpoved". Odpoved ma byt co najkratsia.

Kontext: {ex["context"][:500]}

Otazka: {ex["question"]}

Odpoved:"""
        response = ask(model_name, prompt, max_tokens=50)
        answers = ex["answers"]
        is_correct = False
        if answers and "text" in answers and len(answers["text"]) > 0:
            for ans in answers["text"]:
                if ans.lower() in response.lower() or response.lower() in ans.lower():
                    is_correct = True
                    break
        elif not answers or ("text" in answers and len(answers["text"]) == 0):
            if "ziadna" in response.lower() or response.strip() == "":
                is_correct = True
        if is_correct:
            correct += 1
        total += 1
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{MAX_SAMPLES}] {correct/total*100:.1f}%")
        time.sleep(1)
    acc = correct/total*100
    print(f"  REZULTAT: {acc:.2f}%")
    suffix = model_name.replace("mistral-", "").replace("open-mistral-", "")
    with open(os.path.join(RESULTS_DIR, f"qa_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"task": "question-answering", "model": model_name, "accuracy": round(acc, 2)}, f, indent=2, ensure_ascii=False)
    return acc


print("MISTRAL LARGE")

ml_ner = test_ner("mistral-large-latest", "ner-uner", "ner-uner")
ml_wiki = test_ner("mistral-large-latest", "ner-wikigoldsk", "ner-wikigoldsk")
ml_pos = test_pos("mistral-large-latest")
ml_qa = test_qa("mistral-large-latest")


print("OPEN MISTRAL NEMO")
mn_ner = test_ner("open-mistral-nemo", "ner-uner", "ner-uner")
mn_wiki = test_ner("open-mistral-nemo", "ner-wikigoldsk", "ner-wikigoldsk")
mn_pos = test_pos("open-mistral-nemo")
mn_qa = test_qa("open-mistral-nemo")


print("Suhrn")

print(f"{'Uloha':<25} {'Mistral Large':>15} {'Mistral Nemo':>15}")
print(f"{'ner-uner':<25} {ml_ner:>14.2f}% {mn_ner:>14.2f}%")
print(f"{'ner-wikigoldsk':<25} {ml_wiki:>14.2f}% {mn_wiki:>14.2f}%")
print(f"{'pos':<25} {ml_pos:>14.2f}% {mn_pos:>14.2f}%")
print(f"{'question-answering':<25} {ml_qa:>14.2f}% {mn_qa:>14.2f}%")

