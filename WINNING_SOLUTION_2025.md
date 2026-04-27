# Frist Prize

## Training Data Challenge

- Huge imbalance in taxonomy groups, Non-Bird groups made up 30% but less than 4% of total samples, very noisy non bird groups
- Additional Xeno-Canto data was used, 5489 with target species, 17197 with new species. Max samples per species: 500 for target, any duration, 200 for new species, duration <= 60 sec
- Raw signals normalized by absmax, all secondary labels = 1 , 20 second chunks. 5 seconds don't have long enough context to distinguish signals.

## Models & Training

- Used EfficientNet B0, B3, B4, RegNetY 008, 016, NFNet L0. Gem frequency pooling.
- Cross Entropy loss, cosine annealing warm restarts with warm restart every 5 epochs, total epochs 15. Used mixup augmentation, with equal sampling weight for each species
- Padding strategy for left padding, ensure consistent overlap on the right side.
- LB feedback instead of local CV, ensemble of folds.

## Supervised Training

- Include training data into self training
- Noise injection, strong augmentations of pseudo-labeled samples, model level noise via dropout. (Why Noisy Student, arXiv:1911.04252)
- Mixup training sample with random pseudo labeled sample, stochastic depth enabled self training.
- Adjusting ratio of mixups is very important. The higher ratio, higher the LB score. For this competition, best ratio was 1, meaning each training sample is always mixed with pseudo labeled samples.

## Multi iterative self training

- Training params remained the same except for epochs: 25 drop path rate 0.15 padding strategy random padding, samples shorter than 20 sec were placed at random positions within 20 sec.
- Many papers and solutions from last year showed potential in multi iterative self training.
- Noise reduction via power transform, sampling weights equal to the sum of pseudo labels enabled multi iterative self training.

## Self Training loop

1. Sampler selects with higher prob pseudo-labeled chunks that have higher sum of labels
2. self-training via mixup of labled local data and pseudo-labled soundscape chunks
3. submit / LB feedback
4. Tune power transform to reduce noise
2a. Select the best ensemble from the current iteration and set it as the new teacher
2b. pseudo label train soundscapes

Order: 1 -> 2 -> 3 -> 4 and 2 -> 2a -> 2b -> 4

## Dedicated Models

- Insecta, AMphibia appears almost in all soundscapes
- Training species are very underrepresented, insecta has 155 samples for 17 species, 0.5% of all data, amphibia 583 samples for 34 species, 2.0% of all data, most training samples very noisy.
- United target and additional species are 700, more epochs 40 , larger batch size 128, smaller model efficientnet B0 for dedicated training.
- Train and test soundscapes should be populated mostly with target species, check target species get higher avg pred than extra species.

## Inference

- 1D sliding window segmentation: averages overlapping framewise preds from neighboring audio chunks
- Make use of all preds, acts as a form of test time augmentation.
- Smoothing, delta shift TTA applied
- Final ensemble is 7 models trained on diverse dataset and self training iterations. SI(Self training iteration), ST(Supervised Training)

- 1 efficientnet B4 for 3rd SI, 1 efficientnet B3 from 3rd SI
- 2 RegnetY 016 from 4th SI
- 1 ECA -NFNET -L0 from 3rd SI(with additional Xeno Canto data for target species)
- 1 RegNetY 008 from ST
- 1 efficientnet B0(ST on the extended Amphibia/Insecta species)

- Key for surviving shake up is emsembling, models from different training stages, divers backbone architectures, underrepresented taxonomic groups.

## Key Findings

- Chunk duration should be tailored to the target species
- Self training becomes effective when noise is injected in a domain relevant way
- Multi iterative self training unlocks more value from unlabeled data
- Pseudo-label preprocessing(power transform in this case) and a smart sampling strategy are crucial for effective multi iterative training


# Second Prize

## Validation

- Basic Approach: 5 folds, stratification by primary label, grouping by author.
- Enhancing by 2 options: Addiing undersampled species to validation folds which placed at least one instance per rare species in val, removed from training. Another option is adding all undersampled species to train folds, removing from validation.

## Strong Baseline

- Augmentation to address domain shift: Mixup mixes two waveforms, labels combined via element wise max to mimic multi label data. Background Mixing adds background noise from prior year soundscapes and ESC-50. Spec Augment applies time and freq masking on mel spec. Random filtering is random equalizer to simulate channel distrotions.
- Data sampling strategy: gamma = -0.5 + repeat undersampled classes till 100, gamma = -1
- For backbones, NFNet-L0 or EfficientNetV2-S, label smoothiung with alpha 0.05
- Carefully selected additional data, such as xeno canto dump, CSA dump, INaturalist dump.

## Transfer Learning

- Collected 16607 species from BirdCLEF taxonomies, removed BirdCLEF+ 2025 species, corrupted files, invalid codes, and species with less than 10 recordings, final dataset ~819k recordings, 7489 species.
- Split into 5% holdout, validation on species with more than 100 recordings
- Pretraining trained baseline models without class balancing to learn general patterns.
- Selected best checkpoint by macro ROC AUC, reinitialized classification head.
- Fine tune with same setup as baseline, no changes to learning rates.

## Pseudo Labeling

- Train data fold N: Trained SED model, predict soundscapes wihtout prev iter, pseudo selection logic, selected soundscapes and predicted probs, selected soundscape from previous iterations, removed selected files from prev iter, pseudo train sampling strategy.
- Train data fold N -> pseudo/train sampling strategy -> train model i fold N -> Trained model fold N, predict soundscape fold N, pseudo selection logic, selected soundscapes nad predicted probabilities fold.
- Train sample(class and audio) -> is class present in pseudo dataset? -> if no use train sample, if yes uniform prob > 0.6 and if no use train sample, if yes use pseudo sample.
- For each 5 sec chunk, compute the max pred class prob, discard chunks with max prob 0.5, for retained chunks, set all class prob < 0.1 to zero, keep the remaining soft target vec as the label.
- Mixup hard targets from train audio with soft targets from pseudo dataset via clip(a + b, 0, 1)
- Both models trained on OOF Pseudo and full pseudo were used in ensembling.

## Post processing and ensemble
- Mean: multiply the prob of species in each segment by the average prob of species in the whole audio
- TopN: multiply prob of species in each seg by the average top N prob of the species in the whole audio. Best N = 1 Selected
- Mixing 3 experiments, each containing 5 folds models, 15 models in total.
- SElect 3 best exp with optuna

## Curation
- False positive artifacts in audio samples: Vocalization with alien speech(computer synthesized IDs, spoken descriptions, etc), speech overlapping animal vocalization, human speech as background noise, vocalization with periods of silence.
- These were dealt by apply VAD , manual curation for hearing, apply model prob to find out samples with biggest discrepency
- Sample 5 second random chunk and did this


# Third Prize: semi-supervised learning with soft AUC loss

- Custom soft AUC loss
- Only 2d batch normalization
- 50 hours training time, 10 hours data prep

## Feature Selection and Engineering

- smaller hop-length (64) and larger n_mels (256) mattered only
- human speech filtering
- frequency dropout(not only high, but full freq masking)
- mixup(audio and mel spec)
- 256 x 256 input size

## Training Methods

- SED model/training was more stable
- Custom AUC, sofr AUC loss showed resistant to overfitting
- Semi-supervised learning to make use of un labelled data, 10 efficientnet models as lablers.
- Super sampling of classes with few samples
- 16 efficientnet models as final solution(only efficientnet works, I don't know why but ViT and ResNet didn't work)

soft AUC loss code

```python
class SoftAUCLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super() __init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds, labels, sample_weight = None):
        pos_preds = preds[labels < 0.5]
        neg_preds = preds[labels < 0.5]
        pos_lables = labels[labels > 0.5]
        neg_labels = labels[labels < 0.5]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_weights = torch.ones_like(pos_preds) * self.pos_weight * (pos_labels-0.5)
        neg_weights = torch.ones_like(neg_preds) * self.neg_weight * (0.5 - neg_labels)

        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]* labels.shape[1], dim=1)
            pos_weights = pos_weights * sample_weights
            neg_weights = neg_weights * sample_weights

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0) # [N_pos, N_neg]
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin)) # [Npos, N_neg]

        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)

        return weighted_loss.mean()
```

## Findings

- Small amount of unlabelled data fom past competitions, soft labels improved the model slightly.
- Deleting the predictions (all 0) of some classes with very few samples had little impact on lb score. My models doesn't seem to predict them well.

## Simple Model

- Efficientnet V2 contributes the most
- Remove some large models
- Reduce the size of feature images


