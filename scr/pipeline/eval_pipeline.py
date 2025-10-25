from tokenize import String
from unittest import result
from datasets import load_dataset
from datasets import Dataset
from datasets.formatting import List
from numpy import save
from numpy._core.numeric import dtype
from pyarrow import string
from pyarrow.dataset import dataset
from typing import Optional, List, Dict, Any
from datetime import date, datetime
import os
from pathlib import Path
import json
from tqdm import tqdm


class Eval_pipeline():
    
    def __init__(self, dataset:Dataset, log_folder_path:string = None): 
        
        self.dataset:Dataset = dataset
        
        if log_folder_path is None: 
            self.log_folder_path:string  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
        else:
            self.log_folder_path = log_folder_path
        
        
    def run_eval(self): 
        
        logs = []
        logs_formatted = []
    
        
        
        for sample in tqdm(self.dataset):
            
            result = self.call_agent(question=["Question"] ,system_prompt="You are..." )
            
            
            result_json = {
                "task_id": sample["task_id"],
                "model_answer": result["model_answer"],  
                "reasoning_trace": result["reasoning_trace"]
                
            }
            
            
            log = {
            "Task_id": sample["task_id"],
            "File_name": sample["file_name"],
            "File_path": sample["file_path"],
            "Time_stamp": datetime.now().isoformat(),
            "Question": sample["Question"],
            "Level": sample["Level"],
            "Annotator Metadata": sample["Annotator Metadata"],
            "result_json": result_json
            }

            logs.append(log)
            logs_formatted.append(result_json)
    
    
        self.save_logs(logs_list=logs, logs_list_formatted=logs_formatted)
            
        
            
            
    def call_agent(self, question:str, system_prompt:str) ->dict[str, Any]: 
        
        #TODO: implement logic to call agent
        return {"model_answer": "answer",
                "reasoning_trace": "reasoning"}
                
        
    def
        
    def save_logs(self, logs_list: list[dict[str, Any]], logs_list_formatted: list[dict[str, Any]]) -> None:
        
            file_path_logs = "eval_logs.json"
            file_path_logs_formatted = "results.json"
        
        
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder = Path(self.log_folder_path) / f"eval_{timestamp}"
            os.makedirs(folder, exist_ok=True)

            file_path_logs = Path(folder) / file_path_logs
            file_path_logs_formatted = Path(folder) / file_path_logs_formatted
            
            with open(file_path_logs, "w") as f:
                _ = f.write(json.dumps(logs_list))
            
            with open(file_path_logs_formatted, "w") as f:
                _ = f.write(json.dumps(logs_list_formatted))
            
            
            
            

            
            
    
        
        
#   def get_context(self, filename: string, file_path: string) -> None:
        
    
    


