import argparse
import pandas as pd
parser = argparse.ArgumentParser(description="Example for store true & false")
parser.add_argument("--set-false", "--s", "--sf", help="Set var_name to false", dest="var_name", action="store_false")
parser.set_defaults(var_name=True)

args = parser.parse_args()

# print ("var_name is %s" % args.var_name)
# a  = list(1, 2, 3, 4, 5)
# print(a)

# a = pd.read_csv
print('a            b')