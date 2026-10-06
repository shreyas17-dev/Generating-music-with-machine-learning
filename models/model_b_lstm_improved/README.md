# Model B - improved LSTM

2 layers x 512 hidden units, dropout 0.3, 30 epochs, random transposition of +-2 semitones.
Same data as model A.

Train:  `python models/model_b_lstm_improved/train.py`
Result: `checkpoints/B_lstm_improved_best.pt`.
