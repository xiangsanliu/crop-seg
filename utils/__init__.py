from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--log_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--last_step", type=int, default=0)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--with_eval", type=bool, default=False)
    parser.add_argument("--weight", type=str, default=None)
    return parser.parse_args()
