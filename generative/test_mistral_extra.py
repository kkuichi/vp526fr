import json, os, random, time, re
from datasets import load_dataset
from mistralai import Mistral

API_KEY = "xdsB8Vwt6tkVc8Lx6WajOsmIDIu7uDFV"
MODEL_NAME = "mistral-small-latest"
MAX_SAMPLES = 50  # mensi pocet lebo token-level ulohy su narocnejsie
RESULTS_DIR = "results_generative"
random.seed(42)

client = Mistral(api_key=API_KEY)
print(f"Model: {MODEL_NAME}\n")

def ask(prompt, max_tokens=200, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Retry {attempt+1}... ({e})")
            time.sleep(10)
    return ""


print("ner-uner")


ds_ner = load_dataset("slovak-nlp/sklep", "ner-uner", split="test")
label_names_ner = ds_ner.features["ner_tags"].feature.names

indices = list(range(len(ds_ner)))
random.shuffle(indices)
indices = indices[:MAX_SAMPLES]

correct_tokens = 0
total_tokens = 0
results_ner = []
start = time.time()

for i, idx in enumerate(indices):
    ex = ds_ner[idx]
    tokens = ex["tokens"]
    true_tags = [label_names_ner[t] for t in ex["ner_tags"]]
    
    tokens_str = " ".join(tokens[:50])
    prompt = f"""Pre kazdy token v nasledujucej vete urc NER tag. Pouzi IOB format s tagmi: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.
Odpoved zapis ako zoznam tagov oddelenych medzerami, PRESNE tolko tagov kolko je tokenov.

Tokeny: {tokens_str}

NER tagy:"""
    
    response = ask(prompt, max_tokens=300)
    pred_tags = response.strip().split()
    
    min_len = min(len(true_tags[:50]), len(pred_tags))
    for j in range(min_len):
        if pred_tags[j].upper() == true_tags[j].upper():
            correct_tokens += 1
        total_tokens += 1
    total_tokens += max(0, len(true_tags[:50]) - len(pred_tags))
    
    results_ner.append({"true": true_tags[:50], "pred": pred_tags[:50], "response": response[:200]})
    
    if (i+1) % 10 == 0:
        acc = correct_tokens/total_tokens*100 if total_tokens > 0 else 0
        print(f"  [{i+1}/{MAX_SAMPLES}] Token Accuracy: {acc:.1f}%")
    time.sleep(1)

ner_acc = correct_tokens/total_tokens*100 if total_tokens > 0 else 0
print(f"\n  REZULTAT Token Accuracy: {ner_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")

with open(os.path.join(RESULTS_DIR, "ner-uner_mistral.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "ner-uner", "model": MODEL_NAME, "token_accuracy": round(ner_acc, 2), "predictions": results_ner[:10]}, f, indent=2, ensure_ascii=False)


print("ner-wikigoldsk")


ds_wiki = load_dataset("slovak-nlp/sklep", "ner-wikigoldsk", split="test")
label_names_wiki = ds_wiki.features["ner_tags"].feature.names

indices_w = list(range(len(ds_wiki)))
random.shuffle(indices_w)
indices_w = indices_w[:MAX_SAMPLES]

correct_w = 0
total_w = 0
results_wiki = []
start = time.time()

for i, idx in enumerate(indices_w):
    ex = ds_wiki[idx]
    tokens = ex["tokens"]
    true_tags = [label_names_wiki[t] for t in ex["ner_tags"]]
    
    tokens_str = " ".join(tokens[:50])
    prompt = f"""Pre kazdy token v nasledujucej vete urc NER tag. Pouzi IOB format s tagmi: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.
Odpoved zapis ako zoznam tagov oddelenych medzerami, PRESNE tolko tagov kolko je tokenov.

Tokeny: {tokens_str}

NER tagy:"""
    
    response = ask(prompt, max_tokens=300)
    pred_tags = response.strip().split()
    
    min_len = min(len(true_tags[:50]), len(pred_tags))
    for j in range(min_len):
        if pred_tags[j].upper() == true_tags[j].upper():
            correct_w += 1
        total_w += 1
    total_w += max(0, len(true_tags[:50]) - len(pred_tags))
    
    results_wiki.append({"true": true_tags[:50], "pred": pred_tags[:50], "response": response[:200]})
    
    if (i+1) % 10 == 0:
        acc = correct_w/total_w*100 if total_w > 0 else 0
        print(f"  [{i+1}/{MAX_SAMPLES}] Token Accuracy: {acc:.1f}%")
    time.sleep(1)

wiki_acc = correct_w/total_w*100 if total_w > 0 else 0
print(f"\n  REZULTAT Token Accuracy: {wiki_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")

with open(os.path.join(RESULTS_DIR, "ner-wikigoldsk_mistral.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "ner-wikigoldsk", "model": MODEL_NAME, "token_accuracy": round(wiki_acc, 2), "predictions": results_wiki[:10]}, f, indent=2, ensure_ascii=False)


print("pos")


ds_pos = load_dataset("slovak-nlp/sklep", "pos", split="test")
label_names_pos = ds_pos.features["pos_tags"].feature.names

indices_p = list(range(len(ds_pos)))
random.shuffle(indices_p)
indices_p = indices_p[:MAX_SAMPLES]

correct_p = 0
total_p = 0
results_pos = []
start = time.time()

for i, idx in enumerate(indices_p):
    ex = ds_pos[idx]
    tokens = ex["tokens"]
    true_tags = [label_names_pos[t] for t in ex["pos_tags"]]
    
    tokens_str = " ".join(tokens[:50])
    prompt = f"""Pre kazdy token urc POS (part-of-speech) tag. Pouzi Universal POS tagy: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CONJ, CCONJ, SCONJ, PART, INTJ, AUX, PUNCT, SYM, X, PROPN.
Odpoved zapis ako zoznam tagov oddelenych medzerami, PRESNE tolko tagov kolko je tokenov.

Tokeny: {tokens_str}

POS tagy:"""
    
    response = ask(prompt, max_tokens=300)
    pred_tags = response.strip().split()
    
    min_len = min(len(true_tags[:50]), len(pred_tags))
    for j in range(min_len):
        if pred_tags[j].upper() == true_tags[j].upper():
            correct_p += 1
        total_p += 1
    total_p += max(0, len(true_tags[:50]) - len(pred_tags))
    
    results_pos.append({"true": true_tags[:50], "pred": pred_tags[:50], "response": response[:200]})
    
    if (i+1) % 10 == 0:
        acc = correct_p/total_p*100 if total_p > 0 else 0
        print(f"  [{i+1}/{MAX_SAMPLES}] Token Accuracy: {acc:.1f}%")
    time.sleep(1)

pos_acc = correct_p/total_p*100 if total_p > 0 else 0
print(f"\n  REZULTAT Token Accuracy: {pos_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")

with open(os.path.join(RESULTS_DIR, "pos_mistral.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "pos", "model": MODEL_NAME, "token_accuracy": round(pos_acc, 2), "predictions": results_pos[:10]}, f, indent=2, ensure_ascii=False)



print("question-answering")


ds_qa = load_dataset("slovak-nlp/sklep", "question-answering", split="test")

indices_q = list(range(len(ds_qa)))
random.shuffle(indices_q)
indices_q = indices_q[:MAX_SAMPLES]

correct_q = 0
total_q = 0
results_qa = []
start = time.time()

for i, idx in enumerate(indices_q):
    ex = ds_qa[idx]
    context = ex["context"][:500]
    question = ex["question"]
    answers = ex["answers"]
    
    prompt = f"""Na zaklade kontextu odpoved na otazku. Ak odpoved nie je v texte, napis "ziadna odpoved".
Odpoved ma byt co najkratsia - iba relevantna cast textu.

Kontext: {context}

Otazka: {question}

Odpoved:"""
    
    response = ask(prompt, max_tokens=50)

    is_correct = False
    if answers and "text" in answers and len(answers["text"]) > 0:
        for ans in answers["text"]:
            if ans.lower() in response.lower() or response.lower() in ans.lower():
                is_correct = True
                break
    elif not answers or ("text" in answers and len(answers["text"]) == 0):
        if "ziadna" in response.lower() or "nie" in response.lower() or response.strip() == "":
            is_correct = True
    
    if is_correct:
        correct_q += 1
    total_q += 1
    
    results_qa.append({"question": question, "true": answers.get("text", [])[:3], "pred": response[:100], "correct": is_correct})
    
    if (i+1) % 10 == 0:
        acc = correct_q/total_q*100
        print(f"  [{i+1}/{MAX_SAMPLES}] Accuracy: {acc:.1f}%")
    time.sleep(1)

qa_acc = correct_q/total_q*100
print(f"\n  REZULTAT: {qa_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")

with open(os.path.join(RESULTS_DIR, "question-answering_mistral.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "question-answering", "model": MODEL_NAME, "accuracy": round(qa_acc, 2), "predictions": results_qa[:10]}, f, indent=2, ensure_ascii=False)


print(f"PIDSUIMOK: {MODEL_NAME} (doplnkove ulohy)")

print(f"  ner-uner (token acc)      {ner_acc:.2f}%")
print(f"  ner-wikigoldsk (token acc) {wiki_acc:.2f}%")
print(f"  pos (token acc)           {pos_acc:.2f}%")
print(f"  question-answering        {qa_acc:.2f}%")

