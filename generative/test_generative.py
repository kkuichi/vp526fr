import json
import os
import random
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Milos/slovak-gpt-j-1.4B"
MAX_SAMPLES = 100 
RESULTS_DIR = "results_generative"
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(42)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()
print("nacitany.\n")

def generate(prompt, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

def evaluate_task(task_name, prompt_fn, label_map, parse_fn, dataset_config, 
                  text_field="text", split="test", metric_name="accuracy"):
    print(f"Uloha: {task_name}")
    
    ds = load_dataset("slovak-nlp/sklep", dataset_config, split=split)

    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:MAX_SAMPLES]
    
    correct = 0
    total = 0
    results = []
    
    start_time = time.time()
    
    for i, idx in enumerate(indices):
        example = ds[idx]
        prompt = prompt_fn(example)
        response = generate(prompt, max_new_tokens=15)
        predicted = parse_fn(response, label_map)
        true_label = example["label"]
        
        is_correct = (predicted == true_label)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "idx": idx,
            "true_label": true_label,
            "predicted": predicted,
            "response": response,
            "correct": is_correct,
        })
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            acc = correct / total * 100
            eta = elapsed / (i + 1) * (len(indices) - i - 1)
            print(f"  [{i+1}/{len(indices)}] Accuracy: {acc:.1f}% | ETA: {eta/60:.1f} хв")
    
    elapsed = time.time() - start_time
    accuracy = correct / total * 100
    
    print(f"\n  Vysledok: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Cas: {elapsed/60:.1f} хв")
    
    output = {
        "task": task_name,
        "model": MODEL_NAME,
        "metric": metric_name,
        "value": accuracy,
        "correct": correct,
        "total": total,
        "time_minutes": round(elapsed / 60, 1),
        "predictions": results,
    }
    
    path = os.path.join(RESULTS_DIR, f"{task_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Ulozila: {path}\n")
    
    return accuracy

def parse_classification(response, label_map):
    response_lower = response.lower().strip()
    for key, value in label_map.items():
        if key.lower() in response_lower:
            return value
    return -1  # не може розпізнати

def sentiment_prompt(example):
    text = example["text"][:300]
    return f"""Urc sentiment nasledujuceho textu. Odpoved je: pozitivny, negativny alebo neutralny.

Text: "{text}"
Sentiment:"""

sentiment_labels = {"pozit": 0, "negat": 1, "neutr": 2}
ds_sent = load_dataset("slovak-nlp/sklep", "sentiment-analysis", split="train")
label_names_sent = {name: i for i, name in enumerate(sorted(set(ds_sent["label"])))} if isinstance(ds_sent["label"][0], str) else None

def sentiment_parse(response, label_map):
    r = response.lower()
    if "pozit" in r:
        return 0
    elif "negat" in r:
        return 1
    elif "neutr" in r:
        return 2
    return -1

acc_sent = evaluate_task(
    "sentiment-analysis",
    sentiment_prompt,
    sentiment_labels,
    sentiment_parse,
    "sentiment-analysis",
)

def hate_prompt(example):
    text = example["text"][:300]
    return f"""Je nasledujuci text nenavistny (hate speech)? Odpoved je: ano alebo nie.

Text: "{text}"
Odpoved:"""

def hate_parse(response, label_map):
    r = response.lower()
    if "nie" in r:
        return 0
    elif "ano" in r or "áno" in r:
        return 1
    return -1

acc_hate = evaluate_task(
    "hate-speech",
    hate_prompt,
    {"ano": 1, "nie": 0},
    hate_parse,
    "hate-speech",
)

def nli_prompt(example):
    premise = example["premise"][:200]
    hypothesis = example["hypothesis"][:200]
    return f"""Urc vztah medzi dvoma vetami. Odpoved je: implikacia, rozpor alebo neutralny.

Veta 1: "{premise}"
Veta 2: "{hypothesis}"
Vztah:"""

def nli_parse(response, label_map):
    r = response.lower()
    if "implik" in r or "vyplýv" in r or "entail" in r:
        return 0
    elif "rozp" in r or "contrad" in r:
        return 2
    elif "neutr" in r:
        return 1
    return -1

acc_nli = evaluate_task(
    "nli",
    nli_prompt,
    {"implikacia": 0, "neutralny": 1, "rozpor": 2},
    nli_parse,
    "nli",
)


def rte_prompt(example):
    text1 = example.get("text1", example.get("sentence1", ""))[:200]
    text2 = example.get("text2", example.get("sentence2", ""))[:200]
    return f"""Vyplyva druha veta z prvej? Odpoved je: ano alebo nie.

Veta 1: "{text1}"
Veta 2: "{text2}"
Odpoved:"""

def rte_parse(response, label_map):
    r = response.lower()
    if "nie" in r:
        return 1
    elif "ano" in r or "áno" in r:
        return 0
    return -1

ds_rte = load_dataset("slovak-nlp/sklep", "rte", split="test")


acc_rte = evaluate_task(
    "rte",
    rte_prompt,
    {"ano": 0, "nie": 1},
    rte_parse,
    "rte",
)



print(f"Suhrn: {MODEL_NAME}")

all_results = {
    "sentiment-analysis": acc_sent,
    "hate-speech": acc_hate,
    "nli": acc_nli,
    "rte": acc_rte,
}
for task, acc in all_results.items():
    print(f"  {task:<25} {acc:.2f}%")


summary_path = os.path.join(RESULTS_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "max_samples": MAX_SAMPLES,
        "results": all_results,
    }, f, indent=2, ensure_ascii=False)
print(f"Ulozen: {summary_path}")
