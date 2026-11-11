import argparse
from ..runtime import load_config
from ..trainers.engine import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='Resume explicitly from last.pt in an existing run directory')
    args = parser.parse_args()
    train(load_config(args.config), args.output, args.resume)


if __name__ == '__main__':
    main()
