
import logging
import os, sys, time, csv
from datetime import datetime

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def get_logger(args, name, level=logging.INFO, fmt="%(asctime)s - [%(levelname)s] %(message)s", rank=""):
    logger = logging.getLogger(name)
    # unlike the root logger, a custom logger can’t be configured using basicConfig()
    logging.basicConfig(
        filename = os.path.join(args.out, f"{time_str() if level==logging.INFO else 'dev'}_{rank}.log"),
        format = fmt, datefmt="%Y-%m-%d %H:%M:%S", level = level
    )
    logger.setLevel(level)

    # console print
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

    return logger


class Writer:
    def __init__(self, filename, head=None):
        self.filename = filename
        self.head = head
        # self.register(head)
        
    def register(self, head):
        self.head = head

        with open(self.filename, mode="w", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.head and isinstance(head, (list, tuple)):
                writer.writerow([*self.head, "Time"])
            else:
                pass
    
    def update(self, row):
        with open(self.filename, mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            if isinstance(row, list):
                writer.writerow([*row, datetime.today().strftime('%Y-%m-%d %H:%M:%S')])
            elif isinstance(row, dict):
                writer.writerow([*(row.get(k, "None") for k in self.head), datetime.today().strftime('%Y-%m-%d %H:%M:%S')])
            else:
                pass


def get_writter(args):
    return Writer(os.path.join(args.out, "record.csv"))



