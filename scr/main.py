from pipeline import eval_pipeline
import os
import token
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login




if __name__ == "__main__": 
    
    
    dataset = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        trust_remote_code=True
    )
    
    
    test_data = dataset["test"]
    validation_data = dataset["validation"]
    
    
    evaluation = eval_pipeline.Eval_pipeline(dataset=validation_data)
    
    evaluation.run_eval()
