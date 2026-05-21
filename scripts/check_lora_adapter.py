import json
import sys
from pathlib import Path

adapter_dir = Path(sys.argv[1])
config_path = adapter_dir / "adapter_config.json"
weights_path = adapter_dir / "adapter_model.safetensors"

print(f"adapter_dir: {adapter_dir}")
print(f"adapter_config_exists: {config_path.exists()}")
print(f"adapter_weights_exists: {weights_path.exists()}")

if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print("r:", config.get("r"))
    print("lora_alpha:", config.get("lora_alpha"))
    print("target_modules:", config.get("target_modules"))
    print("base_model_name_or_path:", config.get("base_model_name_or_path"))

if weights_path.exists():
    print(f"adapter_model_size_mb: {weights_path.stat().st_size / 1024 / 1024:.2f}")
