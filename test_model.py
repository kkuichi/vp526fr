from transformers import AutoTokenizer, AutoModel
import torch

model_name = "gerulata/slovakbert"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print(f"Parameters: {model.num_parameters():,}")
    print(f" GPU : {torch.cuda.is_available()}")
    
   
    test_text = "Toto je testovacia veta v slovenčine."
    print(f"\nTest text: '{test_text}'")
    
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    
    print(f"✓ Model successfully processed the text!")
    print(f"✓ Output shape: {outputs.last_hidden_state.shape}")
   
    
except Exception as e:
    print(f"\n✗ Error: {e}")