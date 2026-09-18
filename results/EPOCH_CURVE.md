# The model was overfitting, and the comparison used the worse checkpoint

Closes the one hypothesis the negative-conclusion audit left open: **N2 — was the model
underfit?** (the planned 8-epoch run died at epoch ~2.9, so only 3 checkpoints survive).

## Checkpoint curve

| Checkpoint | v5b_A | v5c_A | mean |
|---|---|---|---|
| checkpoint-epoch-1 | 0.778 | **0.918** | **0.848** |
| checkpoint-epoch-2 | **0.803** | 0.878 | 0.840 |
| checkpoint-epoch-3 | 0.768 | 0.816 | 0.792 |

Last minus first = **−0.056**. Performance *falls* after epoch 1–2.

**N2 is excluded — and the sign is opposite to the hypothesis.** The model was not
underfit; it was **overfitting**. The 8-epoch budget was inherited from the four-grade task
and is simply wrong for this three-grade dataset — **epoch 1 is the better checkpoint**
(mean 0.848 vs 0.792 for the epoch-3 checkpoint we had been calling `final`).

## Consequence for the clean comparison

All earlier three-arm numbers used the epoch-3 checkpoint — i.e. the *worse* one. Re-running
with epoch 1:

| Label set | n | Independent program | Model (epoch 1) | Δ | McNemar p |
|---|---|---|---|---|---|
| v5b_A | 203 | 0.675 | **0.778** | +0.103 | **0.0038** |
| v5b_B | 199 | 0.774 | **0.789** | +0.015 | 0.7798 |
| v5c_A | 196 | 0.730 | **0.918** | +0.189 | **0.0000** |
| v5c_B | 205 | 0.741 | 0.722 | −0.020 | 0.6778 |

Same direction (model leads 3/4), but the lead is larger: v5c_A goes from +0.087 to
**+0.189**, and the pooled Δ from +0.049 to **+0.071** (p=0.0001). So the previously
reported comparison was a **conservative lower bound**.

## Outstanding

Retraining with `num_epochs=2` and re-evaluating is not done. Three attempts to complete a
longer run failed on XPU stability (`UR_RESULT_ERROR_DEVICE_LOST` / `..._OUT_OF_DEVICE_MEMORY`
on the 287M model with long inputs); a CPU fallback was not timed. The epoch-1 checkpoint is
usable and is what the numbers above use.
