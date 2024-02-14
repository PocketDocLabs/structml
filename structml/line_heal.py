import re
import torch
import gc
import os
from rich.progress import track
from multiprocessing import Pool, set_start_method
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# This module is used to fix text with excessive newlines and or ending dashes typical of books and fixed width text such as usenet posts.

# Set logging verbosity to error
logging.set_verbosity_error()

# Default model:
default_model = "Dans-DiscountModels/Dans-StructureEvaluator-Small"

# Length of fake token (used to estimate the length of the text in tokens)
fake_token_length = 5

ram_per_instance = 2.2


def get_gpu_vram():
    # This function returns the available VRAM on the GPU in GB

    # Clear cuda cache
    torch.cuda.empty_cache()

    # Get total VRAM
    total_vram = torch.cuda.get_device_properties(0).total_memory

    # Get available VRAM
    available_vram = total_vram - torch.cuda.memory_reserved(0)

    # Return available VRAM in GB
    return available_vram / (1024 ** 3)


def get_cpu_ram():
    # This function returns the available RAM on the CPU in GB

    # Get available RAM
    available_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

    # Return available RAM in GB
    return available_ram / (1024 ** 3)


def calculate_instances_cuda(safety_margin=0.9):
    # This function calculates the maximum number of instances of a model that can be loaded onto the GPU based on the available VRAM, the VRAM required per instance, and a safety margin

    # Get available VRAM
    available_vram = get_gpu_vram()

    # Calculate maximum instances
    max_instances = int((available_vram * safety_margin) / ram_per_instance)

    # Return maximum instances
    return max_instances


def calculate_instances(safety_margin=0.9):
    # This function calculates the maximum number of instances of a model that can be loaded onto the CPU based on the available RAM, the RAM required per instance, and a safety margin

    # Get available RAM
    available_ram = get_cpu_ram()

    # Calculate maximum instances
    max_instances = int((available_ram * safety_margin) / ram_per_instance)

    # Return maximum instances
    return max_instances


def load_model(model_name=default_model, cuda=True):
    # This function loads a model and tokenizer from Hugging Face's model hub, moves the model to the GPU if available, and returns the model and tokenizer

    if cuda:
        # Set default device to cuda
        torch.set_default_device("cuda")

        # Clear cuda cache
        torch.cuda.empty_cache()

        # Clear CPU cache
        gc.collect()
    else:
        # Set default device to cpu
        torch.set_default_device("cpu")

        # Clear CPU cache
        gc.collect()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Return model and tokenizer
    return model, tokenizer


def check_perplexity(text, model, tokenizer):
 # Encode text to tensor
    input_ids = torch.tensor([tokenizer.encode(text)])

    # Disable dropout (use eval mode)
    model.eval()

    # Calculate loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs[0]

    return torch.exp(loss).item()


def split_text(text):
    # This function splits text at each new line, strips leading and trailing whitespace from each line, and returns the result as a list

    # Merge multiple new lines into one
    text = re.sub(r"\n+", "\n", text)

    # Split text at each new line, store in list
    text = text.split("\n")

    # Strip leading and trailing whitespace from each line
    text = [line.strip() for line in text]

    # Return list
    return text


def trim_string_end(string, trim_amount):
    # This function trims the string to a given amount of characters from the end, to the closest space, and returns the result

    # Trim string
    string = string[:trim_amount]

    # Find the last space in the string
    last_space = string.rfind(" ")

    # If a space was found, trim the string to the last space
    if last_space != -1:
        string = string[:last_space]

    # Return trimmed string
    return string


def trim_string_start(string, trim_amount):
    # This function trims the string to a given amount of characters from the start, to the closest space, and returns the result

    # Trim string
    string = string[-trim_amount:]

    # Find the first space in the string
    first_space = string.find(" ")

    # If a space was found, trim the string to the first space
    if first_space != -1:
        string = string[first_space + 1:]

    # Return trimmed string
    return string


def evaluate_and_join_strings(string1, string2, model, tokenizer):
    # This function checks the perplexity of two strings joined with a list of joining strings, and returns the string with the lowest perplexity

    # Trim string1 to 500 * fake_token_length characters
    trimmed_string1 = trim_string_start(string1, 500 * fake_token_length)

    # Trim string2 to 500 * fake_token_length characters
    trimmed_string2 = trim_string_end(string2, 500 * fake_token_length)

    # List of joining strings
    joiners = [" ", "\n", "\n\n", ""]

    # If string1 ends in a hyphen but not two hyphens, and string2 does not start with a hyphen, remove the hyphen from string1 and add a variant of each joiner with the hyphen prefixed to it to the list of joiners
    if string1.endswith("-") and not string1.endswith("--") and not string2.startswith("-"):
        string1 = string1[:-1]

        joiners += ["-" + joiner for joiner in joiners]

    # Create a dictionary to strings, their joining strings, and their perplexities
    perplexities = []

    # For each joining string
    for joiner in joiners:
        # Join strings
        joined_string = trimmed_string1 + joiner + trimmed_string2

        # Check perplexity of joined string
        perplexity = check_perplexity(joined_string, model, tokenizer)

        # Add joined string and perplexity to dictionary
        perplexities.append((joined_string, perplexity, joiner))

    # Sort perplexities by perplexity with the lowest perplexity first
    perplexities.sort(key=lambda x: x[1])

    # Return the joined string with the lowest perplexity
    return string1 + perplexities[0][2] + string2


def parse(text, verbose=False, cuda=True):
    # This function fixes text with excessive newlines and or ending dashes typical of books and fixed width text such as usenet posts

    # Load model and tokenizer
    model, tokenizer = load_model(cuda=cuda)

    # Split text into lines
    lines = split_text(text)
 
    # Initialize list to store fixed lines
    fixed_text = ""

    # Take the first line and append it to the fixed text
    fixed_text += lines[0]

    # Remove the first line from the list of lines
    lines = lines[1:]

    # For each line, join it with the fixed_text using the joining string with the lowest perplexity
    if verbose:
        for line in track(lines, description="Fixing text", total=len(lines)):
            fixed_text = evaluate_and_join_strings(fixed_text, line, model, tokenizer)
    else:
        for line in lines:
            fixed_text = evaluate_and_join_strings(fixed_text, line, model, tokenizer)

    # Unload model and tokenizer
    del model, tokenizer

    if cuda:
        torch.cuda.empty_cache()

    gc.collect()

    # Return fixed text
    return fixed_text


def parse_with_model(text, model, tokenizer, verbose=False):
    # This function fixes text with excessive newlines and or ending dashes typical of books and fixed width text such as usenet posts

    # Split text into lines
    lines = split_text(text)

    # Initialize list to store fixed lines
    fixed_text = ""

    # Take the first line and append it to the fixed text
    fixed_text += lines[0]

    # Remove the first line from the list of lines
    lines = lines[1:]

    # For each line, join it with the fixed_text using the joining string with the lowest perplexity
    if verbose:
        for line in track(lines, description="Fixing text", total=len(lines)):
            fixed_text = evaluate_and_join_strings(fixed_text, line, model, tokenizer)
    else:
        for line in lines:
            fixed_text = evaluate_and_join_strings(fixed_text, line, model, tokenizer)

    # Return fixed text
    return fixed_text


def parse_list(texts, verbose=False, cuda=True):
    # This function fixes a list of texts with excessive newlines and or ending dashes typical of books and fixed width text such as usenet posts using multiple processess

    # Calculate maximum instances
    max_instances = calculate_instances_cuda() if cuda else calculate_instances()

    # Set start method to spawn (required for multiprocessing with CUDA)
    set_start_method('spawn', force=True)

    # Create a pool of processes with a maximum number of instances
    with Pool(max_instances) as p:
        # For each text in the list, fix it with parse(text)
        if verbose:
            results = list(track(p.imap(parse, texts), description="Fixing texts", total=len(texts)))
        else:
            results = p.map(parse, texts)

    if cuda:
        torch.cuda.empty_cache()

    gc.collect()

    # Return fixed texts
    return results