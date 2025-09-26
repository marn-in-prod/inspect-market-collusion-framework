#!/usr/bin/env python
"""
Script to run the prediction market simulation evaluation.
"""

from inspect_ai import eval
from inspect_ai.model import get_model

from task import create_task_from_json

if __name__ == "__main__":
    # This will work after running the test script to create the config file
    task1 = create_task_from_json("./market-collusion-eval/src/config/minimal_collusion.json")

    results = eval(
        task1,
        model=get_model("openai/gpt-4o-mini", api_key="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")
    )