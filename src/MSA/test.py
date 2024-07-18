import sys
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python test.py <Words>")
        sys.exit(1)

    input_words = sys.argv[1]
    print(input_words)
    df_tem = pd.read_csv(input_words, sep='\t')
    print (len(df_tem))

if __name__ == "__main__":
    main()
