# OLD:
layer2_models = ["nvidia/nemotron-super"] * 4
# ... asyncio.gather call ...

# NEW:
layer2_models = ["nvidia/nemotron-super", "nvidia/deepseek-v3.2", "nvidia/glm-5", "nvidia/qwen-3.5"]
layer2_results = await _race_first_n_debate([(model, layer2_prompt) for model in layer2_models], n=3)