from pathlib import Path
from safetensors.torch import load_file
import torch
import random

class CacheLoader:
    def __init__(self, cache_directory, tokenizer):
        self.cache_directory = Path(cache_directory)
        self.cache_file_paths = sorted(list(self.cache_directory.glob("*.safetensors")))
        self.cache_file_feature_ranges = []
        for cache_file_path in self.cache_file_paths:
            cache_file_feature_range = cache_file_path.stem.split("_")
            self.cache_file_feature_ranges.append({"path": cache_file_path, "start": int(cache_file_feature_range[0]), "end": int(cache_file_feature_range[1])})
        self.tokenizer = tokenizer

    def has_sufficient_activating_examples(self, feature_id, minimum_activating_examples_count):
        cache_file_feature_range = [cache_file_feature_range for cache_file_feature_range in self.cache_file_feature_ranges if cache_file_feature_range["start"] <= feature_id <= cache_file_feature_range["end"]][0]
        cache = load_file(cache_file_feature_range["path"])
        activating_examples_count = (cache["locations"][:, 2] == (feature_id - cache_file_feature_range["start"])).sum().item()
        return activating_examples_count >= minimum_activating_examples_count
    
    def has_sufficient_non_activating_examples(self, feature_id, minimum_non_activating_examples_count):
        cache_file_feature_range = [cache_file_feature_range for cache_file_feature_range in self.cache_file_feature_ranges if cache_file_feature_range["start"] <= feature_id <= cache_file_feature_range["end"]][0]
        cache = load_file(cache_file_feature_range["path"])
        activating_examples_count = (cache["locations"][:, 2] == (feature_id - cache_file_feature_range["start"])).sum().item()
        non_activating_examples_count = cache["tokens"].shape[0] - activating_examples_count
        return non_activating_examples_count >= minimum_non_activating_examples_count
    
    def get_non_activating_examples_split(self, feature_id, k, held_out_set_size):
        cache_file_feature_range, non_activating_examples = self.get_non_activating_examples(feature_id)
        selected_non_activating_examples = non_activating_examples[-k:]
        decoded_selected_non_activating_examples = self.decode_examples(cache_file_feature_range, selected_non_activating_examples)
        if held_out_set_size == 0:
            return decoded_selected_non_activating_examples
        selected_non_activating_example_indices = {example["example_index"] for example in selected_non_activating_examples}
        unselected_non_activating_examples = [example for example in non_activating_examples if example["example_index"] not in selected_non_activating_example_indices]
        non_activating_held_out_set = random.Random().sample(unselected_non_activating_examples, held_out_set_size)
        decoded_non_activating_held_out_set = self.decode_examples(cache_file_feature_range, non_activating_held_out_set)
        return decoded_selected_non_activating_examples, decoded_non_activating_held_out_set
    
    def get_non_activating_examples(self, feature_id):
        cache_file_feature_range = [cache_file_feature_range for cache_file_feature_range in self.cache_file_feature_ranges if cache_file_feature_range["start"] <= feature_id <= cache_file_feature_range["end"]][0]
        cache = load_file(cache_file_feature_range["path"])
        locations = cache["locations"].to(torch.int64)
        feature_mask = locations[:, 2] == (feature_id - cache_file_feature_range["start"])
        example_indices = set(locations[feature_mask, 0].tolist())
        non_activating_examples = []
        for example_index in range(cache["tokens"].shape[0]):
            if example_index not in example_indices:
                non_activating_examples.append({
                    "activation": 0.0,
                    "example_index": example_index,
                })
        return cache_file_feature_range, non_activating_examples

    def get_activating_examples_split(self, feature_id, k, held_out_set_size):
        cache_file_feature_range, activating_examples = self.get_activating_examples(feature_id)
        activating_examples.sort(key=lambda activating_example: activating_example["activation"])
        selected_activating_examples = activating_examples[-k:]
        decoded_selected_activating_examples = self.decode_examples(cache_file_feature_range, selected_activating_examples)
        if held_out_set_size == 0:
            return decoded_selected_activating_examples
        activating_held_out_set = activating_examples[-(k + held_out_set_size):-k]
        decoded_activating_held_out_set = self.decode_examples(cache_file_feature_range, activating_held_out_set)
        return decoded_selected_activating_examples, decoded_activating_held_out_set
    
    def get_activating_examples(self, feature_id):
        cache_file_feature_range = [cache_file_feature_range for cache_file_feature_range in self.cache_file_feature_ranges if cache_file_feature_range["start"] <= feature_id <= cache_file_feature_range["end"]][0]
        cache = load_file(cache_file_feature_range["path"])
        locations = cache["locations"].to(torch.int64) 
        feature_mask = locations[:, 2] == (feature_id - cache_file_feature_range["start"])        
        activations = cache["activations"][feature_mask]
        example_indices = locations[feature_mask, 0]
        activating_examples = []
        for activation, example_index in zip(activations, example_indices):
            activating_examples.append({
                "activation": activation.item(),
                "example_index": example_index.item()
            })
        return cache_file_feature_range, activating_examples

    def decode_examples(self, cache_file_feature_range, examples):
        cache = load_file(cache_file_feature_range["path"])
        decoded_examples = []
        for example in examples:
            example_tokens = cache["tokens"][example["example_index"]]
            example_raw_text = self.tokenizer.decode([example_token for example_token in example_tokens if example_token != self.tokenizer.pad_token_id], skip_special_tokens=False)
            example_text = self.remove_text_tags(example_raw_text)
            decoded_examples.append({
                "activation": example["activation"],
                "text": example_text,
            })
        return decoded_examples
    
    def remove_text_tags(self, raw_text):
        raw_text = raw_text.replace("<|begin_of_text|>", "")
        raw_text = raw_text.replace("<|start_header_id|>", "\n\n")
        raw_text = raw_text.replace("<|end_header_id|>", "")
        raw_text = raw_text.replace("<|eot_id|>", "") 
        return raw_text.strip()