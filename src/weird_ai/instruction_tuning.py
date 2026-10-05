"""
Instruction fine-tuning helpers for Weird AI.
"""

import torch


def extract_response(generated_text, prompt_text):
    """
    Remove the prompt from generated text and return only the response.
    """

    # TODO:
    # Remove prompt_text from the beginning of generated_text.
    # Strip extra whitespace.

    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):]
    return generated_text.replace("### Response:", "").strip()

def save_instruction_model(model, path):
    """
    Save instruction fine-tuned model weights.
    """

    # TODO:
    # Use torch.save with model.state_dict().

    torch.save(model.state_dict(), path)


def load_instruction_model(model, path, device):
    """
    Load instruction fine-tuned model weights.
    """

    # TODO:
    # Use torch.load and model.load_state_dict.

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
