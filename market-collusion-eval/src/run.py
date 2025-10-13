#!/usr/bin/env python
"""
Script to run the prediction market simulation evaluation with Together AI.
"""

from inspect_ai import eval
from inspect_ai.model import get_model

from task import create_task_from_json

if __name__ == "__main__":
    task1 = create_task_from_json("config/01_baseline.json")

    results = eval(
        task1,
        # Replace with the TogetherAI model you want to use
        model=get_model(
            "openai/gpt-4",  
            api_key=""
        )
    )
