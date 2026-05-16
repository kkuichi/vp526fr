import os
import subprocess
import json
from datetime import datetime

MODEL_NAME = "gerulata/slovakbert"

TASKS_CONFIG = {

    "sentiment-analysis": {
        "script": "run_classification.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "sentiment-analysis",
            "text_column_names": "text",  
        }
    },
    "hate-speech": {
        "script": "run_classification.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "hate-speech",
            "text_column_names": "text",  
        }
    },

    "nli": {
        "script": "run_glue.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "nli",
            "task_name": "mnli",
            "max_train_samples": "20000",  
        }
    },
    "sts": {
        "script": "run_glue.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "sts",
            "task_name": "stsb",
        }
    },
    "rte": {
        "script": "run_glue.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "rte",
            "task_name": "rte",
        }
    },


    "question-answering": {
        "script": "run_qa.py",
        "max_seq_length": "384",
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "question-answering",
            "doc_stride": "128",
            "max_train_samples": "10000",
            "max_eval_samples": "3000",
            "max_predict_samples": "3000",
        }
    },


    "ner-wikigoldsk": {
        "script": "run_ner.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "ner-wikigoldsk",
            "text_column_name": "tokens",
            "label_column_name": "ner_tags",
        }
    },
    "ner-uner": {
        "script": "run_ner.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "ner-uner",
            "text_column_name": "tokens",
            "label_column_name": "ner_tags",
        }
    },
    "pos": {
        "script": "run_ner.py",
        "params": {
            "dataset_name": "slovak-nlp/sklep",
            "dataset_config_name": "pos",
            "text_column_name": "tokens",
            "label_column_name": "pos_tags",
        }
    },
}


def build_command(task_name, config):
    script = config["script"]
    params = config["params"]

    output_dir = f"./results/{task_name}_{MODEL_NAME.replace('/', '_')}"

    cmd = [
        "venv\\Scripts\\python.exe",
        f"eval/scripts/{script}",
        "--model_name_or_path", MODEL_NAME,
        "--do_train",
        "--do_eval",
        "--do_predict",                       
        "--max_seq_length", str(config.get("max_seq_length", "128")),
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "3",             
        "--output_dir", output_dir,
        "--logging_strategy", "steps",
        "--logging_steps", "50",               
        "--eval_strategy", "epoch",            
        "--save_strategy", "epoch",            
        "--load_best_model_at_end", "True", 
        "--metric_for_best_model", config.get("metric_for_best_model", "eval_loss"),
    ]

    if config.get("greater_is_better"):
            cmd.extend(["--greater_is_better", "True"])


    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    return cmd


def run_task(task_name, config):
    
    print(f"Uloha: {task_name} | Skript: {config['script']} | Epochy: 3")
    

    cmd = build_command(task_name, config)

    
    output_dir = f"./results/{task_name}_{MODEL_NAME.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\n[OK] {task_name} - uspesne dokoncene\n")
            return {"task": task_name, "status": "SUCCESS", "error": None}
        else:
            print(f"\n[CHYBA] {task_name} - navratovy kod: {result.returncode}\n")
            return {"task": task_name, "status": "FAILED", "error": f"Exit code: {result.returncode}"}
    except Exception as e:
        print(f"\n[CHYBA] {task_name} - {e}\n")
        return {"task": task_name, "status": "FAILED", "error": str(e)}

def main():
    import sys
    print(f"Doladenie modelu {MODEL_NAME} na benchmarku SKLEP")
    print(f"Epochy: 3 | Eval: validation + test | Logovanie: kazdych 50 krokov")
    
    if len(sys.argv) > 1:
        tasks_to_run = sys.argv[1:]
        print(f"\nSpustam ulohy: {', '.join(tasks_to_run)}")
    else:
        tasks_to_run = list(TASKS_CONFIG.keys())
        print(f"\nSpustam vsetky ulohy: {', '.join(tasks_to_run)}")

    results = []

    for task_name in tasks_to_run:
        if task_name not in TASKS_CONFIG:
            print(f"\nUloha '{task_name}' nie je v konfiguracii, preskakujem.")
            continue
        config = TASKS_CONFIG[task_name]
        result = run_task(task_name, config)
        results.append(result)

    
    
    for r in results:
        status = "OK" if r["status"] == "SUCCESS" else "CHYBA"
        print(f"  {r['task']:<25} {status}")

    success = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"\nUspesne: {success}/{len(results)}")
    

    with open("results_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "epochs": 3,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print("Vysledky ulozene v: results_summary.json")


if __name__ == "__main__":
    main()