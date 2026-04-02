import asyncio
from openai import PermissionDeniedError

class FeatureLabeler:
    def __init__(self, cache_loader, held_out_set_size, k_positive, k_zero, labeling_instructions, client, prediction_instructions, async_client):
        self.cache_loader = cache_loader
        self.k_positive = k_positive
        self.k_zero = k_zero
        self.held_out_set_size = held_out_set_size
        self.labeling_instructions = labeling_instructions
        self.client = client
        self.prediction_instructions = prediction_instructions
        self.async_client = async_client

        self.feature_descriptions = {}
        self.feature_description_evaluations = {}

    def get_feature_description(self, selected_activating_examples, selected_non_activating_examples):
        labeling_prompt = self.labeling_instructions
        for i, sample in enumerate(reversed(selected_activating_examples)):
            labeling_prompt += f"\nPositively scored conversation {i + 1}:\n"
            labeling_prompt += f"{sample['text']}\n"
            labeling_prompt += "-----"    
        for i, sample in enumerate(selected_non_activating_examples):
            labeling_prompt += f"\nZero scored conversation {i + 1}:\n"
            labeling_prompt += f"{sample['text']}\n"
            labeling_prompt += "-----"
        response = self.client.responses.create(
            model="gpt-5-nano",
            reasoning={"effort": "medium"},
            input=labeling_prompt
        )
        return response.output_text

    async def predict_feature_activation(self, held_out_example, prediction_prompt, semaphore):
        prediction_prompt += f"\nConversation:\n"
        prediction_prompt += f"{held_out_example['text']}\n"
        async with semaphore:
            try:
                response = await self.async_client.responses.create(
                    model="gpt-5-nano",
                    reasoning={"effort": "medium"},
                    input=prediction_prompt
                )
                prediction = "Positive" if "Positive" in response.output_text else "Zero" if "Zero" in response.output_text else None
                if prediction is None:
                    print(f"Unparseable prediction: {response.output_text}")
                    return None
                return prediction == ("Positive" if held_out_example["activation"] > 0 else "Zero")
            except PermissionDeniedError:
                print(f"Permission denied during prediction")
                return None

    async def evaluate_feature_description(self, feature_description, held_out_set):
        prediction_prompt = self.prediction_instructions
        prediction_prompt += "\nFeature Description:\n"
        prediction_prompt += f"{feature_description}\n"
        semaphore = asyncio.Semaphore(50) 
        tasks = [self.predict_feature_activation(held_out_example, prediction_prompt, semaphore) for held_out_example in held_out_set]
        correct_example_mask = await asyncio.gather(*tasks)
        correct_example_mask = [example for example in correct_example_mask if example is not None]
        return {
            "accuracy": sum(correct_example_mask) / len(correct_example_mask),
            "correct_count": sum(correct_example_mask),
            "total_count": len(correct_example_mask),
            "feature_description": feature_description,
        }

    async def process_feature(self, feature_id):
        if not self.cache_loader.has_sufficient_activating_examples(feature_id, self.k_positive + self.held_out_set_size) or not self.cache_loader.has_sufficient_non_activating_examples(feature_id, self.k_zero + self.held_out_set_size):
            print(f"Skipping feature {feature_id}: insufficient examples.")
            return
        selected_activating_examples, activating_held_out_set = self.cache_loader.get_activating_examples_split(feature_id, self.k_positive, self.held_out_set_size)
        selected_non_activating_examples, non_activating_held_out_set = self.cache_loader.get_non_activating_examples_split(feature_id, self.k_zero, self.held_out_set_size)
        print(f"Generating description for feature {feature_id}...")
        try:
            feature_description = self.get_feature_description(selected_activating_examples, selected_non_activating_examples)
            self.feature_descriptions[feature_id] = feature_description
        except PermissionDeniedError:
            print(f"Skipping feature {feature_id}: permission error.")
            return
        print(f"Evaluating description for feature {feature_id}...")
        self.feature_description_evaluations[feature_id] = await self.evaluate_feature_description(feature_description, activating_held_out_set + non_activating_held_out_set)
