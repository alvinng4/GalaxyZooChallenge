# GalaxyZooChallenge
Late submission attempt for Galaxy Zoo - The Galaxy Challenge on Kaggle.

This is my Final Year Project (Sem 1) PHYS4610 at CUHK. The report is provided in the repository.

Highlights:
* Constructed a CNN baseline model from scratch using the PyTorch framework
* Averaging multiple outputs by exploiting rotational and reflectional invariances greatly improve the score
* Baseline + CBAM (Convolutional Block Attention Module) reproduced single-model score of competition winner
* Transfer learning greatly improves the score and reduces training time
* Pretrained DINOv2 gave the best score
* Pretrained ResNet50 has the shortest runtime and a great performance

Best single-model score:

| Model            | Score (RMSE)    | # of Epochs | Pretrained dataset | Total runtime / per Epoch (s) |
|------------------|----------|-------------|--------------------|-------------------------------|
| Baseline         | 0.07742  | 240         | /                  | 19200 / 80                    |
| Baseline + CBAM  | 0.07693  | 240         | /                  | 19680 / 82                    |
| ResNet50 (not pretrained)        | 0.07574  | 120         | /                  | 15000 / 125                   |
| ResNet50         | 0.07393  | 30          | ImageNet-1K v1     | 3750 / 125                    |
| ResNet101        | 0.07400  | 30          | ImageNet-1K v1     | 5520 / 184                    |
| ConvNeXt Large   | 0.07357  | 30          | ImageNet-1K v1     | 45000 / 1500                  |
| DINOv2 Base      | 0.07251  | 15          | LVD-142M           | 17400 / 580                   |
