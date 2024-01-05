import os
import torch

from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from omegaconf import DictConfig
from transformers import AutoTokenizer
from importlib import import_module

class GenshinRequest(BaseModel):
    claim: str
    para: str

class GenshinAPI(FastAPI):
    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()

        self.post("/api/predict")(self.predict)
    
    def predict(self, req: GenshinRequest) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []

        return output