import argparse

parser = argparse.ArgumentParser(description="Greet someone.")
parser.add_argument("--name", default="world", help="who to greet")
parser.add_argument("--times", type=int, default=1, help="repetitions")

args = parser.parse_args()          # lit sys.argv
print(args.name, args.times)        # accès par attribut
