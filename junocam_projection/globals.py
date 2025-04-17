import signal

FRAME_HEIGHT = 128
FRAME_WIDTH = 1648


NC_FOLDER = "./nc/"
MOS_FOLDER = "./mos/"
MASK_FOLDER = "./mask/"
NPY_FOLDER = "./npy/"


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
