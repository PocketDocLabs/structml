[![structml Logo](https://github.com/PocketDocLabs/structml/blob/main/assets/logo/logo_2x_scale.png?raw=true)]()

[![PyPI](https://img.shields.io/pypi/v/structml.svg)](https://pypi.org/project/structml/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/structml.svg)](https://pypi.org/project/structml/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/structml)](https://pypistats.org/packages/structml)

# structml
**Not ready for production use**

A package for working with data using NLP and ML.


## Features
- The primary feature of this package is the `line_heal` module, which provides functions to reconstruct the original continuous text from a string that has been broken into multiple lines.
- Parallel processing for processing multiple strings at once.
- CPU and CUDA support.


## Installation
**Requirements**: Python >=3.9 and Pytorch >=2.0.

To install the package from PyPI, run the following command:
```bash
pip install structml
```

To install the package from source, run the following command:
```bash
git clone https://github.com/PocketDocLabs/structml.git
cd structml
pip install .
```


## Usage

### line_heal
The `line_heal` module provides functions to reconstruct the original continuous text from a string that has been broken into multiple lines. This is done by choosing the most probable way to join lines or paragraphs, and remove hyphens if appropriate. The best fitting option is chosen based on the perplexity of the text, using a language model.

#### Functions
##### `parse`
`parse` is the main function of the `line_heal` module. It takes a string of text and returns the healed string.
```python
def parse(string: str, verbose: bool = False, cuda: bool = True) -> str:
    """
    Heal a broken line of text.

    Args:
        string (str): The broken string of text.
        verbose (bool): Whether to print debug information.
        cuda (bool): Whether to use the GPU for processing.

    Returns:
        str: The healed line of text.
    """
```

###### Example
```python
from structml import line_heal

line = "This line is broken into mul-\ntiple lines as is common in data\nthat originates from a structured\nsource.\nThe goal of this function is to\nheal the line in a way flows\nnaturally and is easy to read."

healed_line = line_heal.parse(line)

print(healed_line)
```
###### Output
```
This line is broken into multiple lines as is common in data that originates from a structured source.

The goal of this function is to heal the line in a way flows naturally and is easy to read.
```

---

##### `parse_list`
**Note**: This function uses large amounts of memory.
`parse_list` is a function that takes a list of strings and returns a list of healed strings.
Uses multiple processes to speed up the healing process. 
```python
def parse_list(strings: List[str], verbose: bool = False, cuda: bool = True) -> List[str]:
    """
    Heal a list of broken lines of text.

    Args:
        strings (List[str]): The list of broken strings of text.
        verbose (bool): Whether to print debug information.
        cuda (bool): Whether to use the GPU for processing.

    Returns:
        List[str]: The list of healed lines of text.
    """
```

###### Example
```python
from structml import line_heal

lines = [
    "This line is broken into mul-\ntiple lines as is common in data\nthat originates from a structured\nsource.\nThe goal of this function is to\nheal the line in a way flows\nnaturally and is easy to read.",
    "Another goal of this function is to\ndo so in a way that is efficient\nand scalable."
]

healed_lines = line_heal.parse_list(lines)

for line in healed_lines:
    print(line)
```
###### Output
```
This line is broken into multiple lines as is common in data that originates from a structured source.

The goal of this function is to heal the line in a way flows naturally and is easy to read.
Another goal of this function is to do so in a way that is efficient and scalable.
```


## Credits
- Thank you to [Sebastian Gabarain](https://huggingface.co/Locutusque) for [the model](https://huggingface.co/Locutusque/TinyMistral-248M-v2.5) that Dans-StructureEvaluator is based on.
- This project was largely inspired by [this project](https://github.com/pd3f/dehyphen) from which the method for healing lines was adapted.
- The training for this project would not have been possible without [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl).

## Model pages
- [Dans-StructureEvaluator](https://huggingface.co/Dans-DiscountModels/Dans-StructureEvaluator-Small), used in the `line_heal` module. Used to check the perplexity of the text with various joining strategies.


## License
This project is licensed under the terms of the AGPL-3.0 license. The full text of the license can be found in the `LICENSE` file.
