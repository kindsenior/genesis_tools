#!/usr/bin/env python3

import argparse

from sbr1_env import Sbr1Env
from genesis_tools.exporter import dump_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    args = parser.parse_args()

    dump_policy(
        env_class=Sbr1Env,
        log_dir=args.log_dir,
        ckpt=args.ckpt,
        env_kwargs={"show_viewer": True},
    )


if __name__ == "__main__":
    main()