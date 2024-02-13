import re
import torch
import gc
from rich.progress import track
import os
from multiprocessing import pool

from transformers import AutoModelForCausalLM, AutoTokenizer

# This module is used to fix text with excessive newlines and or ending dashes typical of books and fixed width text such as usenet posts.

# Default model:
default_model = "Dans-DiscountModels/Dans-StructureEvaluator-Small"

# Length of fake token (used to estimate the length of the text in tokens)
fake_token_length = 5

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


def tokenize_text(text, tokenizer):
    # This function tokenizes text using a given tokenizer and returns the tokenized text

    # Tokenize text
    tokenized_text = tokenizer.encode(text, return_tensors="pt")

    return tokenized_text


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

    del model, tokenizer

    # Unload model and tokenizer
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

def parse_list(text_list, verbose=False, cuda=True):
    # This function fixes a list of texts with excessive newlines and or ending dashes typical of books and fixed width text such as usenet posts

    # Load model and tokenizer
    model, tokenizer = load_model(cuda=cuda)

    # Initialize list to store fixed texts
    fixed_texts = []

    # For each text in the list, fix it and append it to the fixed_texts list
    if verbose:
        for text in track(text_list, description="Fixing texts", total=len(text_list)):
            fixed_texts.append(parse(text, verbose=False, cuda=cuda))
    else:
        for text in text_list:
            fixed_texts.append(parse(text, verbose=False, cuda=cuda))

    # Unload model and tokenizer
    del model, tokenizer

    if cuda:
        torch.cuda.empty_cache()

    gc.collect()

    # Return fixed texts
    return fixed_texts