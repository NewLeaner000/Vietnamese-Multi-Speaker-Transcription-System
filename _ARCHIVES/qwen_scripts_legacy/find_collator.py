import importlib
import inspect
import sys
sys.stdout.reconfigure(encoding='utf-8')
import trl
for name, obj in inspect.getmembers(trl):
    if 'Collator' in name:
        print(f"trl.{name}")
