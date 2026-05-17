# Solution

## Reproducibility instructions

Same as in initial guide:

Install packages:
```bash
pip install -r requirements.txt
```

Run evaluation:

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 32 \
    --output results.json
```

## Final solution description

Modified components:

- `head_init.py`: implements ridge-regression head initialization from frozen
  ResNet18 features.
Other files were kept unchanged.

The head initialization is the most important part of the solution. Instead of
starting classifier with random weights, `head_init.py` first
uses the pretrained frozen ResNet18 backbone to turn CIFAR100 training images into
embeddings. Then it solves a ridge-regression problem to choose
weights and biases for the new 100-class classifier. These values are copied
into `fc.weight` and `fc.bias` before fine-tuning.

Final metrics:

| Checkpoint | Top-1 accuracy |
|---|---:|
| Baseline (ImageNet head) | 0.37% |
| Initialized head (no fine-tuning) | 61.43% |
| Fine-tuned (ZO) | 61.44% |

The evaluation uses `32 * 32 = 1024` fine-tuning samples, which is below the
budget. Most of the gain comes before this loop: CIFAR100 train samples are
passed once through the frozen backbone inside `head_init.py`, and the resulting
features are used to solve the 100-class head in closed form. This works better
than starting from random weights because the pretrained backbone already gives
strong image embeddings, while the zero-order loop has too few steps. The remaining ZO step still tunes `fc.weight` and `fc.bias`, but the change from 61.43% to 61.44% is small enough
to treat as noise. Other SPSA variants also did not add useful profit after the
strong ridge head.

Main implementation details:

- The pretrained ResNet18 backbone is used only as a frozen feature extractor.
  Each image is represented by a 512-dimensional embedding.
- Use the same transforms as during validation.
- The target matrix is a smoothed one-hot encoding.
- The ridge system is solved for the augmented matrix with regularization term equal to 1000 for weights.

The experiments below explain why I removed more complicated ZO optimization
techniques: they helped weaker heads, but did not improve the strong ridge head.

## Experiments and failed attempts

I organized experiments as research questions. Each group was motivated by the
failure or limitation of the previous one.

### RQ1: What is the baseline quality?

Hypothesis: the original ImageNet classifier head should not transfer directly
to CIFAR100, and a randomly initialized CIFAR100 head should also be weak before
fine-tuning.

| Experiment | Result |
|---|---:|
| Initial baseline checkpoints | 0.37 / 1.21 / 1.21 |

The baseline confirmed that the final classifier head is the main bottleneck.
This motivated RQ2: better initialization of the replacement CIFAR100 head.

### RQ2: Do standard random initialization strategies help?

Hypothesis: standard initialization schemes may change optimization stability,
but they probably cannot solve the semantic mismatch between ImageNet features
and CIFAR100 labels.

| Initialization | Result | 
|---|---:|
| Xavier | 1.36 |
| Orthogonal | 0.90 |
| Small-scale random | 1.45 |
| Learned / prior bias | 1.45 |

These methods stayed close to initial accuracy. The reason is that they do not
use the pretrained representation in a class-aware way. This motivated RQ3:
reuse semantic information already present in the ImageNet pretrained model.

### RQ3: Can ImageNet semantic transfer initialize the CIFAR100 head?

Hypothesis: related ImageNet classifier rows can provide useful directions for
CIFAR100 classes, for example transferring animal
directions from the pretrained ImageNet head.

| Initialization | Result |
|---|---:|
| Baselime | 1.03 |
| Copy related ImageNet class weights | 21.57 |
| Copy ImageNet class weights with logit calibration | 21.50 |
| Copy ImageNet class weights with label smoothing | 22.47 |

Semantic transfer worked much better than random initialization. Label
smoothing improved it slightly by making weak label mappings less noisy. This
showed that the frozen backbone already contains useful features, but the manual
mapping from CIFAR100 labels to ImageNet labels is imperfect. 

### RQ4: Can zero-order optimization improve a semantic-transfer head?

Hypothesis: after semantic transfer, the head is useful but still imperfect, so
SPSA updates should be able to improve it.

| Optimizer setting | Result |
|---|---:|
| Baseline | 22.47 |
| SPSA | 50.10 |

SPSA worked well in this setting. The semantic-transfer head provided a useful
starting point, but it was still weak enough that updates
could improve it. This motivated keeping ZO optimization, but also
suggested that the biggest gains came from better head information.

### RQ5: Does data augmentation contribute much under this budget?

Hypothesis: augmentation may improve generalization, but with only a small
number of ZO steps it may be less important than head initialization.

| Augmentation | Observation |
|---|---|
| Resize + horizontal flip | Kept as the active conservative option. |
| `color_erasing` | Did not produce a meaningful improvement. |
| `autoaugment` | Did not produce a meaningful improvement. |

The main improvement came from head initialization, not augmentation. This
motivated focusing the remaining experiments on better ways to construct the
head.

### RQ6: Can sample-based frozen-backbone features replace manual semantics?

Hypothesis: instead of manually matching CIFAR100 labels to ImageNet labels,
pass CIFAR100 training images through the frozen backbone and initialize the
head from the result.

| Method | Result |
|---|---:|
| Head initilized from images passed through frozen backbone | 53.01 |

This worked better than semantic transfer. It uses real CIFAR100 samples and the
actual frozen feature space, so it avoids noisy manual label matching. However,
after this stronger initialization, SPSA stopped giving consistent extra
profit: full-head random perturbations often damaged useful class directions. This motivated RQ7: use a closed-form classifier.

### RQ7: Can ridge regression produce a better closed-form head?

Hypothesis: class centroids are useful, but ridge regression should be better
because it fits all classes jointly and learns both weights and biases for the
frozen-backbone feature space.

| Method | Result |
|---|---:|
| Ridge-regression head | 61.43 |

This became the final selected method. It uses the same frozen ResNet18
features, but solves a regularized linear classification
problem rather than assigning only normalized class centroids. I tried several
ridge regularization values, including `1000`, `2000`, and `3000`; the current
implementation uses `l2 = 1000.0`. In the current code, the remaining ZO
optimizer still updates `fc.weight` and `fc.bias`, but the final result shows
that almost all of the accuracy comes from the ridge-initialized head.
