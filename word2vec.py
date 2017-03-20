# -*- coding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from matplotlib import font_manager, rc

import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


## this code is related to korean word2vec for mac environment
font_fname = '/Library/Fonts/AppleGothic.ttf'     # A font of your choice
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
print font_manager.get_fontconfig_fonts()


# Sample sentences
sentences = [
      "북한이 이날 발사한 SLBM은 현재까지 북한이 진행한 4차례의 시험발사 중에서 가장 먼 500㎞를 비행, 북한이 수중사출 기술에 이어 비행기술까지 상당 수준 확보한 것으로 분석됐다",
      "북한은 이번에 SLBM을 정상보다 높은 각도로 발사해 의도적으로 사거리를 줄였으며 정상각도였다면 사거리가 1천㎞ 이상인 것으로 추정됐다",
      "북한의 잠수함 능력이 향상돼 1천㎞ 이상을 이동, SLBM을 발사한다면 북한에서 3천500㎞ 떨어진 괌까지도 타격할 수 있는 셈이다.",
      "북한이 SLBM을 시험 발사한 것은 이번이 4번째다. 지난해 11월 첫 시험 때는 아예 수면 밖으로 솟구치지도 못했지만 2번째 시험이었던 지난 4월 23일에는 수심 10여ｍ에 있던 잠수함에서 발사된 SLBM이 물 밖으로 솟아올라 약 30㎞를 비행했다",
      "북한의 SLBM 발사는 모든 탄도미사일 발사를 금지하고 있는 유엔 안전보장이사회 결의 위반으로, 정부는 미국, 일본 등 우방국들과 안보리 차원의 대응 방안을 협의할 예정이다.",
            ]
# 'sentences' is 'list'

class Word2Vec():


    #Configuration
    batch_size = 20
    embedding_size = 2 # This is just for visualization
    num_sampled = 15 # Number of negative examples to sample

    def __init__(self, batch_size =20, embedding_size=2,num_sampled=15):
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.num_sampled=num_sampled

        self.RDIC=self.Initialization(sentences)
        self.Model()


    def Initialization(self,sentences):
        # words: list of all words
        words = " ".join(sentences).split()
        #print "'words' is %s and length is %d." %(type(words), len(words))
        # count: list of pairs, each pair consists of 'cats', 10
        count = collections.Counter(words).most_common()
        #print "'count' is %s and length is %d." %(type(count), len(count))
        #print "Word count of top five is %s" %(count[:5])

        # build dictionary
        # count : the group of (word, frequency)...
        # rdic(reverse_dictionary) : extract only word in count
        rdic = [i[0] for i in count] # reverse dic, idx -> word
        '''
        enumerate example:  for i, name  in enumerate(['body','foo','bar'])
                                  print(i,name)
                            0 boby
                            1 foo
                            2 bar
        '''
        dic = {w: i for i, w in enumerate(rdic)} # dic, word -> id
        self.voc_size = len(dic)
        #print "'rdic' is %s and length is %d." % (type(rdic), len(rdic))
        #print "'dic' is %s and length is %d." % (type(dic), len(dic))

        # Make indexed wor data
        data = [dic[word] for word in words]

        #print "'data' is %s and length is %d." % (type(data), len(data))
        #print 'Sample data: numbers: %s / words : %s' %(data[:10], [rdic[t] for t in data[:10]])

        # let's make a training data for window size 1 for simplicity
        # ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox)

        cbow_pairs = []
        for i in range(1, len(data) - 1):
            cbow_pairs.append([[data[i-1], data[i+1]], data[i]])

        #print 'Context pairs: %s' % (cbow_pairs[:10])
        #print "'chow_pairs' is %s and length is %d" % (type(cbow_pairs), len(cbow_pairs))

        # let's make a skip gram
        # the quick brown fox jumps over the lazy dog
        # (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
        self.skip_gram_pairs = []
        for c in cbow_pairs:
            self.skip_gram_pairs.append([c[1], c[0][0]])
            self.skip_gram_pairs.append([c[1], c[0][1]])

        #print "'skip_Gram_pairs' is %s and length is %d." % (type(skip_gram_pairs), len(skip_gram_pairs))
        #print 'skip-gram pairs', skip_gram_pairs[:5]

        return rdic


    def generate_batch(self, size):
        assert size < len(self.skip_gram_pairs)
        x_data = []
        y_data = []
        r = np.random.choice(range(len(self.skip_gram_pairs)), size, replace=False)
        for i in r:
            x_data.append(self.skip_gram_pairs[i][0]) # n dim
            y_data.append([self.skip_gram_pairs[i][1]]) # n, 1dim
        return x_data, y_data


    def Model(self):
        # construct network

        # input dta
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        # need to shpae [batch_size, 1] for nn.nce_loss
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])


        # Look up embeddings for inputs
        self.embeddings = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs) # lookup table

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size], -1.0, 1.0))
        nce_biases = tf.Variable(tf.zeros([self.voc_size]))


        _loss=tf.nn.nce_loss(nce_weights, nce_biases, self.train_labels,  embed, self.num_sampled, self.voc_size)
        # Compute the average NCE loss for the batch
        self.loss = tf.reduce_mean(_loss)

        # Use the adam optimizer
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        print "Network ready"

    def RUN(self):

        # Launch the graph in a session
        with tf.Session() as sess:
            # initializing all variables
            tf.initialize_all_variables().run()

            for step in range(3000):
                batch_inputs, batch_labels = self.generate_batch(self.batch_size)
                _, loss_val = sess.run([self.train_op, self.loss], feed_dict={self.train_inputs: batch_inputs, self.train_labels: batch_labels})

                if step % 500 == 0:
                    print ("Loss at %d : %.5f" %(step, loss_val))

            trained_embeddings = self.embeddings.eval()

        sess.close()

        return trained_embeddings





if __name__=="__main__":

    WV=Word2Vec()
    trained_embeddings=WV.RUN()

    # Show word2vec if dim is 2
    if trained_embeddings.shape[1] == 2:
        labels = WV.RDIC[:20] # show top 20 words
        for i, label in enumerate(labels):
            x, y = trained_embeddings[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
        plt.show()

    pass

