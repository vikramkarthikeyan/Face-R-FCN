import pandas as pd
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv("losses.csv", header=0, index_col=0)

    plt.figure()
    plt.plot(df["Batch"], df["Loss"], color='r')
    plt.xlabel("Batches")
    plt.ylabel("Total Loss")
    plt.title("Total Loss")
    plt.savefig("Total Loss.jpg")

    plt.figure()
    plt.plot(df["Batch"], df["RPN Classification Loss"], color='b')
    plt.xlabel("Batches")
    plt.ylabel("RPN Classification Loss")
    plt.title("RPN Classification Loss")
    plt.savefig("RPN Classification Loss.jpg")

    plt.figure()
    plt.plot(df["Batch"], df["RPN Regression Loss"], color='g')
    plt.xlabel("Batches")
    plt.ylabel("RPN Regression Loss")
    plt.title("RPN Regression Loss")
    plt.savefig("RPN Regression Loss.jpg")

    plt.figure()
    plt.plot(df["Batch"], df["RCNN Classification Loss"], color='c')
    plt.xlabel("Batches")
    plt.ylabel("RCNN Classification Loss")
    plt.title("RCNN Classification Loss")
    plt.savefig("RCNN Classification Loss.jpg")

    plt.figure()
    plt.plot(df["Batch"], df["RCNN Regression Loss"], color='m')
    plt.xlabel("Batches")
    plt.ylabel("RCNN Regression Loss")
    plt.title("RCNN Regression Loss")
    plt.savefig("RCNN Regression Loss.jpg")

    # Batch,Loss,RPN Classification Loss,RPN Regression Loss,RCNN Classification Loss,RCNN Regression Loss


if __name__ == '__main__':
    main()