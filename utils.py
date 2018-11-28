import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import random


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file1 = os.path.join(data_dir, "data1.npy")
        tensor_file2 = os.path.join(data_dir, "data2.npy")
        tensor_file3 = os.path.join(data_dir, "data3.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file1) and os.path.exists(
                tensor_file2) and os.path.exists(tensor_file3)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file1, tensor_file2, tensor_file3)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file1, tensor_file2, tensor_file3)
        self.create_batches()
        self.reset_batch_pointer()

    # preprocess data for the first time.
    def preprocess(self, input_file, vocab_file, tensor_file1, tensor_file2, tensor_file3):
        file = open(input_file)
        in_data = []
        y_data = []
        dict_map = {}
        while 1:
            lines = file.readlines(100000)
            if not lines:
                break
            for line in lines:
                i = 0
                mid = -1
                while i < len(line):
                    if line[i] == '\t':
                        if mid == -1:
                            temp_a = line[:i]
                            mid = i
                        else:
                            temp_b = line[mid + 1:i]
                            mid = i
                    if line[i] in dict_map:
                        dict_map[line[i]] += 1
                    else:
                        dict_map[line[i]] = 1

                    if line[i] == "\n":
                        temp_t = line[mid + 1:i]
                    i += 1
                in_data.append((temp_a, temp_b))
                y_data.append(temp_t)

        count_pairs = sorted(dict_map.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(1, len(self.chars) + 1)))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        # print("self.vocab.get is ", self.vocab.get)
        # print("data is ", data)
        result1 = []
        result2 = []
        result3 = []

        print("seq_length is ", self.seq_length)
        for i, x in enumerate(in_data):
            temp_a = list(map(self.vocab.get, x[0]))
            temp_b = list(map(self.vocab.get, x[1]))

            if len(temp_a) < self.seq_length:
                while len(temp_a) < self.seq_length:
                    temp_a.append(0)
            else:
                temp_a = temp_a[:self.seq_length]

            if len(temp_b) < self.seq_length:
                while len(temp_b) < self.seq_length:
                    temp_b.append(0)
            else:
                temp_b = temp_b[:self.seq_length]

            result1.append(temp_a)
            result2.append(temp_b)
            result3.append(int(y_data[i]))

        self.tensor1 = np.array(result1)
        self.tensor2 = np.array(result2)
        self.tensor3 = np.array(result3)

        np.save(tensor_file1, self.tensor1)
        np.save(tensor_file2, self.tensor2)
        np.save(tensor_file3, self.tensor3)

    # load the preprocessed the data if the data has been processed before.
    def load_preprocessed(self, vocab_file, tensor_file1, tensor_file2, tensor_file3):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        self.tensor1 = np.load(tensor_file1)
        self.tensor2 = np.load(tensor_file2)
        self.tensor3 = np.load(tensor_file3)

        self.num_batches = int(len(self.tensor1) / (self.batch_size))

    # seperate the whole data into different batches.
    def create_batches(self):
        self.num_batches = int(len(self.tensor1) / self.batch_size)

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # reshape the original data into the length self.num_batches * self.batch_size * self.seq_length for convenience.
        self.tensor1 = self.tensor1[:self.num_batches * self.batch_size]
        self.tensor2 = self.tensor2[:self.num_batches * self.batch_size]
        self.tensor3 = self.tensor3[:self.num_batches * self.batch_size]
        xdata1 = self.tensor1
        xdata2 = self.tensor2
        ydata = self.tensor3

        # 3×50×10

        self.x1_batches = np.split(xdata1,
                                   self.num_batches, 0)
        self.x2_batches = np.split(xdata2,
                                   self.num_batches, 0)

        self.y_batches = np.split(ydata,
                                  self.num_batches, 0)

    def next_batch(self):
        x1, x2, y = self.x1_batches[self.pointer_list[self.pointer]], self.x2_batches[self.pointer_list[self.pointer]], \
                    self.y_batches[self.pointer_list[self.pointer]]
        self.pointer += 1
        return x1, x2, y

    def reset_batch_pointer(self):
        temp = list(range(0, self.num_batches))
        random.shuffle(temp)
        self.pointer_list = temp
        self.pointer = self.pointer_list[0]
