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
[makemore](https://github.com/karpathy/makemore)). Training takes a few minutes
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
