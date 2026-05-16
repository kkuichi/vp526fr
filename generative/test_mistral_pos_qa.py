import json, os, random, time
from datasets import load_dataset
from mistralai import Mistral

API_KEY = "xdsB8Vwt6tkVc8Lx6WajOsmIDIu7uDFV"
MODEL_NAME = "mistral-small-latest"
MAX_SAMPLES = 50
RESULTS_DIR = "results_generative"
random.seed(42)

client = Mistral(api_key=API_KEY)
print(f"Model: {MODEL_NAME}\n")

def ask(prompt, max_tokens=200, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Retry {attempt+1}... ({e})")
            time.sleep(10)
    return ""


print("pos")


ds_pos = load_dataset("slovak-nlp/sklep", "pos", split="test")
indices = list(range(len(ds_pos)))
random.shuffle(indices)
indices = indices[:MAX_SAMPLES]

correct = 0
total = 0
results = []
start = time.time()

for i, idx in enumerate(indices):
    ex = ds_pos[idx]
    tokens = ex["tokens"][:50]
    true_tags = ex["pos_tags"][:50]
    
    tokens_str = " ".join(tokens)
    prompt = f"""Pre kazdy token urc POS (part-of-speech) tag. Pouzi Universal POS tagy: NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CCONJ, SCONJ, PART, INTJ, AUX, PUNCT, SYM, X, PROPN.
Odpoved zapis ako zoznam tagov oddelenych medzerami, PRESNE tolko tagov kolko je tokenov ({len(tokens)} tagov).

Tokeny: {tokens_str}

POS tagy:"""
    
    response = ask(prompt, max_tokens=300)
    pred_tags = response.strip().split()
    
    min_len = min(len(true_tags), len(pred_tags))
    for j in range(min_len):
        if pred_tags[j].upper() == true_tags[j].upper():
            correct += 1
        total += 1
    total += max(0, len(true_tags) - len(pred_tags))
    
    results.append({"true": true_tags, "pred": pred_tags[:len(true_tags)]})
    
    if (i+1) % 10 == 0:
        acc = correct/total*100 if total > 0 else 0
        print(f"  [{i+1}/{MAX_SAMPLES}] Token Accuracy: {acc:.1f}%")
    time.sleep(1)

pos_acc = correct/total*100 if total > 0 else 0
print(f"\n  REZULTAT Token Accuracy: {pos_acc:.2f}%")
print(f"  Cas: {(time.time()-start)/60:.1f} min\n")

with open(os.path.join(RESULTS_DIR, "pos_mistral.json"), "w", encoding="utf-8") as f:
    json.dump({"task": "pos", "model": MODEL_NAME, "token_accuracy": round(pos_acc, 2)}, f, indent=2, ensure_ascii=False)

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
    json.dump({"task": "question-answering", "model": MODEL_NAME, "accuracy": round(qa_acc, 2)}, f, indent=2, ensure_ascii=False)


print(f"  pos (token acc)        {pos_acc:.2f}%")
print(f"  question-answering     {qa_acc:.2f}%")

