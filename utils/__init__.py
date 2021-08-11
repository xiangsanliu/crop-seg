from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Training")
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-weight", type=str, required=False)
    return parser.parse_args()
