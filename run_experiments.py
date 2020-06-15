import argparse
import os
from pathlib import Path


def main(args):
    configs = sorted(Path(args.config_directory).glob('*'))
    for config in configs:
        command = f"python posthoc.py {config}"
        print(command)
        os.system(command)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("config_directory", help="directory with configs.")

    args = parser.parse_args()

    main(args)
