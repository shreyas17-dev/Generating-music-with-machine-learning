# Model A - paper-style LSTM

1 layer, 256 hidden units, dropout 0.1, 20 epochs, no augmentation.
Trained on 300 MAESTRO v3 pieces (40 validation, 40 test), 1024-token windows.

Train:  `python models/model_a_paper_lstm/train.py`
Result: `checkpoints/A_paper_style_best.pt` (lowest validation loss).
