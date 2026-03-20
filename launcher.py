import subprocess
import sys
import os

"""
This file spawns one terminal for each model
The purpose is to run the models in parallel to speed things up
"""

MODELS = [
    'ClaudeSonnet4-6',
    'ClaudeHaiku4-5',
    'GPT5-2',
    'GPT4-1',
    'DeepSeekReasoner',
    'DeepSeekChat',
    'Grok4-1_NonReasoning',
    'Grok3Mini',
]

def main():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')

    for model in MODELS:
        cmd = f'start "Experiment: {model}" cmd /c "{sys.executable} {script_path} --model {model}'
        print(f"[LAUNCHER] Starting {model}...")
        subprocess.Popen(cmd, shell=True)

    print(f"[LAUNCHER] Spawned {len(MODELS)} terminals.")

if __name__ == "__main__":
    main()