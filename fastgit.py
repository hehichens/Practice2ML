import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--add", help="add file path or dir", default="./*")
parser.add_argument("--comment", help="comment", default=":fire:update")
args = parser.parse_args()

if __name__ == "__main__":
    os.system("git add {}".format(args.add))
    os.system("git commit -m {}".format(args.comment))
    os.system("git push")
