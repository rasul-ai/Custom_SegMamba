# Swin UNETR baseline (BraTS2020), for comparison with SegMamba

This folder trains and evaluates [Swin UNETR](https://arxiv.org/abs/2201.01266) using the same pipeline as SegMamba. The comparison stays controlled because both models share:

| | SegMamba (`3_train.py`) | Swin UNETR (this folder) |
|---|---|---|
| Data | `data/fullres/train` | same |
| Split | `get_train_val_test_loader_from_train` (70/10/20, seed 42) | same |
| Patch / batch | 128³ / 2 | same |
| Augmentation | `light_training` nnU-Net-style | same |
| Budget | 1000 epochs × 250 steps, validation every 2 epochs | same |
| Loss | 4-class cross entropy | same (`loss_name = "ce"`) |
| Optimizer | SGD lr 1e-2, poly | AdamW lr 1e-4, warmup + cosine (`optimizer_name = "sgd"` for SegMamba's) |
| Inference | sliding window 0.5 overlap, gaussian, 8× mirror TTA | same |
| Metrics | `5_compute_metrics.py` (Dice + HD95 for TC/WT/ET) | same script |

All settings are in `config.py`.

## Usage (run on the GPU machine, from the repo root)

```bash
# 0. optional: check peak GPU memory for the configured batch/patch size (takes about a minute)
python swinunetr/check_gpu_memory.py

# 1. train  -> logs/swinunetr/ (tensorboard + checkpoints + data_split.json)
python swinunetr/train.py

# 2. predict the test split -> prediction_results/swinunetr/
python swinunetr/predict.py --ckpt logs/swinunetr/model/best_model_0.7493.pt

# 3. metrics, using the same script as SegMamba
python 5_compute_metrics.py --pred_name swinunetr
python 5_compute_metrics.py --pred_name segmamba

# 4. side-by-side table with a paired Wilcoxon test
python swinunetr/compare_models.py
```

## GPU memory (RTX 4090, 24 GB)

Swin UNETR at 128³ with feature_size 48 is memory-hungry. Without gradient checkpointing, batch size 2 is likely to exceed 24 GB. `config.py` therefore sets
`use_checkpoint = True`, which recomputes activations in the backward pass and costs roughly 20–30 % more time per step. It also uses AMP (from the shared trainer) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `train.py` prints the peak memory after every validation.
If `check_gpu_memory.py` still reports that memory is tight, set `batch_size = 1`. For inference, lower `sw_batch_size` to 1.

Note: Gradient checkpointing saves memory by discarding the intermediate results of the forward pass and recomputing them later, when the backward pass needs them.

Why training uses so much memory. To compute gradients, the backward pass needs the intermediate outputs ("activations") of every layer from the forward pass. Take y = softmax(q·kᵀ): computing the gradient of softmax requires its output y. So by default PyTorch keeps every layer's activations in memory from the start of the forward pass until the backward pass reaches that layer. At the peak, just before backward begins, all of them are held at once. For a 3D network on 128³ patches, those activations use far more memory than the weights: roughly 15 of your 16.5 GB, against about 1 GB for the model and optimizer.

What checkpointing changes:

Forward pass: a checkpointed block runs as normal. PyTorch keeps only its input and throws away everything computed inside it, such as the Q/K/V tensors, attention maps, softmax outputs and MLP hidden layers.
Backward pass: when the gradient reaches that block, PyTorch runs the block's forward computation again from the saved input to regenerate those activations. It then computes the block's gradients and frees them straight away.
The difference in what stays in memory:


without:  [input][qkv][attn][softmax][mlp_hidden]...   ← kept for every block
with:     [input]                                      ← kept per block
          + one block's internals at a time, during its backward
So memory for the checkpointed parts goes from "all blocks' internals" down to "all blocks' inputs plus one block's internals".

The cost is compute, not accuracy. Each checkpointed block runs its forward pass twice, which adds roughly 20–30% to the time per step. The gradients are the same as without checkpointing, apart from tiny floating-point differences, so the trained model is no worse.

In Swin UNETR specifically. MONAI wraps each Swin transformer block when use_checkpoint=True (monai/networks/nets/swin_unetr.py):


for blk in self.blocks:
    if self.use_checkpoint:
        x = checkpoint.checkpoint(blk, x, mask_matrix, use_reentrant=False)
    else:
        x = blk(x, mask_matrix)
Those blocks are the right thing to checkpoint because window attention creates large attention matrices, about 4 GB per block at the first stage for your batch size. The convolutional encoder and decoder are not checkpointed, so their full-resolution feature maps still account for most of the 16.5 GB you measured.

Why accuracy is unaffected. The recomputed activations are the same numbers the first forward pass produced: same weights, same input, same operations. So the gradients and weight updates match those of a run without checkpointing. 

So, it saves memory by not storing activations. But they're recomputed during the backward pass, block by block, not at the final layer.



## Notes

- **The split must match SegMamba's.** The split shuffles `glob()` output, and file order can differ between machines. Train on the same machine and with the same
  `data/fullres/train` folder that SegMamba used. `train.py` saves the split to `logs/swinunetr/data_split.json`, and `predict.py` refuses to run if the test set differs.
- Swin UNETR is trained from scratch, like SegMamba. No pretrained weights are used.
