# microgpt

A complete GPT — training and inference — in ~200 lines of pure, dependency-free
Python. From [Andrej Karpathy's gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95):

> The most atomic way to train and run inference for a GPT in pure, dependency-free
> Python. This file is the complete algorithm. Everything else is just efficiency.

No NumPy, no PyTorch — just `os`, `math`, and `random`. It trains a tiny
character-level transformer on a list of ~32k human names and then samples new,
made-up names from it.

## Run it

```bash
python3 microgpt.py
```

On first run it downloads the dataset (`input.txt`, the names list from
[makemore](https://github.com/karpathy/makemore) - raw file [here](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt)). Training takes a few minutes
since every scalar multiply is a Python object operation. Expected output:

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.6497
--- inference (new, hallucinated names) ---
sample  1: kamon
sample  2: ann
...
```

## How the file is organized

The script reads top to bottom as the full recipe:

1. **Dataset** — reads `input.txt` into a list of documents (one name per line)
   and shuffles them.

2. **Tokenizer** — character-level: each unique character in the dataset becomes
   a token id, plus one special `BOS` (beginning-of-sequence) token used to mark
   the start and end of every name. Vocab size is 27 (26 letters + BOS).

3. **Autograd (`Value` class)** — a scalar-valued reverse-mode autodiff engine
   (same idea as [micrograd](https://github.com/karpathy/micrograd)). Every
   number in the network is a `Value` node that remembers its children and local
   derivatives; `loss.backward()` topologically sorts the computation graph and
   applies the chain rule to fill in every parameter's `.grad`.

4. **Parameters** — plain Python lists of lists of `Value`s: token embeddings
   (`wte`), position embeddings (`wpe`), per-layer attention weights
   (`wq/wk/wv/wo`), MLP weights (`fc1/fc2`), and the output head (`lm_head`).
   4,192 parameters total.

5. **Model (`gpt` function)** — GPT-2 architecture with minor simplifications:
   RMSNorm instead of LayerNorm, ReLU instead of GELU, no biases. It processes
   one token at a time and appends each position's keys/values to a KV cache,
   which is how causal attention falls out naturally: position `t` can only
   attend to the keys/values cached so far.

6. **Training loop** — 1,000 steps. Each step takes one name, wraps it in `BOS`
   tokens, forwards every position to get next-character probabilities, averages
   the cross-entropy loss over the sequence, backprops, and applies an **Adam**
   update with linear learning-rate decay.

7. **Inference** — starts from `BOS` and repeatedly samples the next character
   from the softmax (at temperature 0.5) until the model emits `BOS` again,
   yielding 20 new name-like words that don't exist in the dataset.

## Explain Like I'm Five (ELI5)

Here is a simple, easy-to-understand breakdown of the key concepts and terms used in `microgpt`:

### 📊 Training Stats
* **`num params` (Number of Parameters)**: The model's "memory knobs" or dials. It is the total count of adjustable numbers the AI uses to learn patterns.
* **`step`**: A single practice round or training iteration. The model looks at one name, makes a prediction, and adjusts its dials.
* **`loss`**: The "wrongness score." It measures how bad the model's guesses are. The goal is to make this number as small as possible.

### ⚙️ The Autograd Engine (How the model learns)
* **`Autograd`**: The automatic mistake tracker. It works out how changing each knob in the brain affects the final wrongness score.
* **`scalar-valued reverse-mode autodiff engine`**: A system that traces backward through the math, one single number (scalar) at a time, to calculate how to adjust each knob.
* **`children and local derivatives`**: Each number remembers its parent math steps (its "children") and how a tiny nudge to itself affects the next direct step (its "local derivative").
* **`loss.backward()`**: The trigger command that says, "Okay, trace all math steps backward from the final loss score to find who to blame for the incorrect predictions."
* **`topologically sorts the computation graph`**: Ordering all math steps in perfect reverse order (from last to first) so we calculate the blame for the later steps before the earlier ones.
* **`applies the chain rule to fill in every parameter's .grad`**: Multiplying local derivatives step-by-step (like a chain reaction) to calculate the final `grad` (gradient/blame value) for every single parameter knob.

### 🧱 Parameters & Embeddings (The pieces of the brain)
* **`Parameters`**: The list of all 4,192 adjustable numbers (knobs) that the model tweaks during training to learn.
* **`embeddings`**: Turning letters into coordinates on a "meaning map" so that letters with similar roles end up close to each other.
* **`token embeddings (wte)`**: A lookup table of coordinates for each unique letter (token) in the alphabet.
* **`position embeddings (wpe)`**: A lookup table of coordinates representing where a letter is located in the word (e.g., first letter vs. fifth letter).
* **`per-layer attention weights (wq/wk/wv/wo)`**: The knobs that help the model focus on other letters in the name (using Queries, Keys, Values, and Outputs) to guess what comes next.
* **`MLP weights (fc1/fc2)`**: The "thinking layers" (Multi-Layer Perceptron) that process all the collected information and decide on the next character.
* **`output head (lm_head)`**: The final projection layer that maps the model's internal thoughts back into actual predictions/probabilities for the 27 possible letters.
* **`4,192 parameters total`**: The total count of all adjustable knobs across the entire model.

### 🧠 Model Architecture & GPT-2 (The structure)
* **`Model`**: The overall neural network machine (functions and weights) that converts letters into next-letter predictions.
* **`GPT-2 architecture`**: The specific blueprint for how embeddings, self-attention, and MLP blocks are wired together, identical to OpenAI's GPT-2 but scaled down.
* **`RMSNorm` vs `LayerNorm`**: Clean-up steps that keep numbers from growing too large or small. `RMSNorm` (Root Mean Square Normalization) does this slightly faster than traditional `LayerNorm` by skipping centering (making the mean zero).
* **`ReLU` vs `GELU`**: "On/Off" switches (activation functions) that allow the model to make complex choices. `ReLU` is a sharp switch (makes negative values zero), while `GELU` is a smoother version of this switch.
* **`biases`**: Extra offsets added to the weights, omitted here (no biases) to keep the code as clean and simple as possible.
* **`KV cache`**: A notepad where the model writes down the attention keys (K) and values (V) of letters it has already seen, so it doesn't waste time recalculating them.
* **`causal attention falls out naturally: position t can only attend to the keys/values cached so far`**: Since the model only writes down keys and values for letters it has already read, it physically cannot look ahead to see or cheat on future letters.

### 🔄 Training Loop & Optimizer
* **`steps`**: Individual training rounds where the model reads a name and learns from it.
* **`forwards every position to get next-character probabilities`**: Running the characters forward through the model to predict what letter comes next after every character.
* **`averages the cross-entropy loss over the sequence`**: Measuring how wrong the predictions were for all characters in the name and calculating the average score.
* **`backprops`**: Doing backward propagation (running `loss.backward()`) to calculate how to adjust each knob.
* **`applies an Adam update with linear learning-rate decay`**: Adjusting the knobs using the Adam optimizer, starting with large adjustments and slowly making them smaller (linear decay) as the model gets smarter.

### 🎲 Softmax & Temperature
* **`softmax`**: A mathematical function that converts raw scores into percentages/probabilities that add up to 100%.
* **`temperature 0.5`**: The "creativity dial." A temperature of `0.5` makes the model more confident and safe in its letter choices, whereas higher values (closer to `1.0`) make the output more random and diverse.

### 📐 Model Configuration & Hyperparameters
* **`Layers (n_layer)`**: How many transformer processing blocks are stacked together (1 layer).
* **`Embedding dim (n_embd)`**: The length of the coordinate list representing each letter's meaning (16 numbers).
* **`Attention heads (n_head)`**: The number of parallel attention views the model has to look at different letters simultaneously (4 heads).
* **`Context length (block_size)`**: The maximum length of a word/sequence the model can read at one time (16 characters).
* **`Vocab size`**: The number of unique tokens the model knows (27: the 26 letters of the alphabet plus a start/end marker).
* **`Parameters`**: The 4,192 adjustable knobs.
* **`Optimizer`**: The training assistant that updates the parameters based on the calculated gradients.
* **`Adam`**: A smart optimizer that adjusts the learning rate for each knob individually.
* **`lr 0.01 with linear decay`**: The initial step size for updates (0.01), which shrinks linearly to 0 by the end of training.

## Model configuration

| Hyperparameter | Value |
|---|---|
| Layers (`n_layer`) | 1 |
| Embedding dim (`n_embd`) | 16 |
| Attention heads (`n_head`) | 4 |
| Context length (`block_size`) | 16 |
| Vocab size | 27 |
| Parameters | 4,192 |
| Optimizer | Adam, lr 0.01 with linear decay |
| Training steps | 1,000 (one document per step) |

## Files

- `microgpt.py` — the whole thing.
- `input.txt` — the names dataset; auto-downloaded on first run (gitignored).
- `run.log` — output of a training run (gitignored).
