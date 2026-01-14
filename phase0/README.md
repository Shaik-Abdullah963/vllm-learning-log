
### Logits Shape and Next-Token Selection

The transformer outputs logits for every position in the input sequence, so the raw shape is:

\[
\text{logits} \in \mathbb{R}^{B \times T \times V}
\]

where:

- \( B \) = batch size  
- \( T \) = sequence length (number of input tokens)  
- \( V \) = vocabulary size  

For next-token generation, we only use the logits of the last time step:

```python
next_token_logits = logits[:, -1, :]
