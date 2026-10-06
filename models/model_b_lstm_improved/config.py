# Model B: improved LSTM, 2 layers x 512, dropout 0.3, transposition augmentation
NAME = "B_lstm_improved"
CONFIG = dict(cell="lstm", hid=512, layers=2, drop=0.3, epochs=30, augment=True)
