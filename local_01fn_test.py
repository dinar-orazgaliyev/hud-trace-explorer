"""Local test script for the trace-explorer environment.

Usage:
    python local_test.py
    python local_test.py --trace-id <uuid>
"""

import asyncio
import json
import os

import hud

from env import env
from hud.agents.claude import ClaudeAgent
from hud.settings import settings
import argparse

from local_test import HUD_API_KEY

HUD_API_KEY = settings.api_key or os.environ.get("HUD_API_KEY", "")

async def test_coding_task_false_negative(trace_id:str):
    task = env(
        "coding_task_false_negative_analysis",
        trace_id = trace_id,
        hud_api_key = HUD_API_KEY,
        query='',
        ground_truth=None,
    )

    async with hud.eval(task) as ctx:
        agent = ClaudeAgent.create(model="claude-sonnet-4-5")
        result = await agent.run(ctx, max_steps=50)
        print(f"Done: {result.done}, Reward: {result.reward}")


async def main(trace_id):
    if not HUD_API_KEY:
        print("ERROR: HUD_API_KEY not set. Set via environment or hud.ai settings.")
        return

    await test_coding_task_false_negative(trace_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trace-explorer environment.")
    parser.add_argument("--trace-id", type=str, default='d36cc3a0-195a-4bf3-b125-4784c7bfa3ea', help="The trace ID to test.")
    args = parser.parse_args()

    asyncio.run(main(args.trace_id))