# Model A: paper-style, 1-layer LSTM, 256 hidden units
NAME = "A_paper_style"
CONFIG = dict(cell="lstm", hid=256, layers=1, drop=0.1, epochs=20, augment=False)
