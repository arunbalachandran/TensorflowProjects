import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
# number_lines_to_process = 1000000  # hm_lines

def all_words_extractor(positive_text, negative_text):
    list_words = []
    for fi in [positive_text, negative_text]:
        with open(fi, 'r') as f:
            for line in f.readlines():
                list_words += list(word_tokenize(line.lower()))
    list_words = [lemmatizer.lemmatize(i) for i in list_words]
    word_counts = Counter(list_words)
    # now limit the number of words
    limited_words = [word for word in word_counts if word_counts[word] < 1000 and word_counts[word] > 50]
    return limited_words

def classification_helper(input_file, list_words, classification):
    feature_set = [] # feature_set for training
    with open(input_file, 'r') as f:
        for line in f.readlines():
            words = [lemmatizer.lemmatize(i) for i in word_tokenize(line.lower())]
            # now for each line, make a one hot vector
            temp = np.zeros(len(list_words))
            for word in words:
                if word.lower() in list_words:
                    temp[list_words.index(word.lower())] += 1
            feature_set.append([list(temp), classification]) 
    return feature_set
     
# now we use the positive and negative sample texts as the sample texts for generating training data
# we are now testing the file on 10% of the data set
def classification_feature_creator(positive_text, negative_text, test_size=0.1):
    list_words = all_words_extractor(positive_text, negative_text)
    feature_vectors = []
    feature_vectors += classification_helper(positive_text, list_words, [1, 0])
    feature_vectors += classification_helper(negative_text, list_words, [0, 1])
    random.shuffle(feature_vectors)
    feature_vectors = np.array(feature_vectors)
    testing_size = int(test_size*len(feature_vectors))  # get the size of the 10%
    training_input = feature_vectors[:, 0][:-testing_size]  # take advantage of numpy slicing
    training_output = feature_vectors[:, 1][:-testing_size]
    test_input = feature_vectors[:, 0][-testing_size:]
    test_output = feature_vectors[:, 1][-testing_size:]
    return training_input, training_output, test_input, test_output

if __name__ == '__main__':
    training_input, training_output, test_input, test_output = classification_feature_creator('pos.txt', 'neg.txt')
    with open('sent_set.pickle', 'wb') as f:
        pickle.dump([training_input, training_output, test_input, test_output], f)
