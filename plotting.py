import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(filepath_or_buffer="losses/losses0.csv",
                   header=0,
                   index_col=0
                   )

for i in range(1, 10):
    data = data.append(other=pd.read_csv(filepath_or_buffer="losses/losses{}.csv".format(i),
                                         header=0,
                                         index_col=0),
                       ignore_index=True
                       )

x = data.shape[0]

plt.figure()
plt.plot(data['Loss'], 'r')
plt.ylabel('Loss')
plt.xlabel('Samples')
plt.title('Loss')

plt.figure()
plt.plot(data['RPN Classification Loss'], 'r')
plt.ylabel('RPN Classification Loss')
plt.xlabel('Samples')
plt.title('RPN Classification Loss')

plt.figure()
plt.plot(data['RPN Regression Loss'], 'r')
plt.ylabel('RPN Regression Loss')
plt.xlabel('Samples')
plt.title('RPN Regression Loss')

plt.figure()
plt.plot(data['RCNN Classification Loss'], 'r')
plt.ylabel('RCNN Classification Loss')
plt.xlabel('Samples')
plt.title('RCNN Classification Loss')

plt.figure()
plt.plot(data['RCNN Regression Loss'], 'r')
plt.ylabel('RCNN Regression Loss')
plt.xlabel('Samples')
plt.title('RCNN Regression Loss')

# ['Batch', 'Loss', 'RPN Classification Loss', 'RPN Regression Loss', 'RCNN Classification Loss', 'RCNN Regression Loss']

# fig, axs = plt.subplots(3, 1, constrained_layout=True)
# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# fig.suptitle('Generator Losses', fontsize=16)
# axs[0].plot(data[''] , 'b')
# axs[0].set_xlabel('Batches')
# axs[0].set_ylabel('Generator Adversarial Loss')
# axs[0].set_title('Generator Adversarial Loss')
#
# axs[1].plot(data[''], 'r')
# axs[1].set_xlabel('Batches')
# axs[1].set_ylabel('Generator Content Loss')
# axs[1].set_title('Generator Content Loss')

# axs[2].plot(data['Generator Loss'], 'g')
# axs[2].set_xlabel('Batches')
# axs[2].set_ylabel('Generator Total Loss')
# axs[2].set_title('Generator Total Loss')

plt.show()
