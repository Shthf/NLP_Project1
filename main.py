from UnigramModel import *
from BigramModel import *


train_path = "./train-Spring2024.txt"
test_path = "./test.txt"
sentence_path = "./sentence.txt"

# train_path = "./mini_train.txt"
# test_path = "./mini_test.txt"
# sentence_path = "./mini_sentence.txt"

output = open("./output.txt", 'w')

def addTags(file_name):
    """
    Pad each sentence in the training and test corpora with start<s> and end</s> symbols

    Args:
        file_name: a string of file path

    Returns:
        file.read(): a string of file after padding
    """
    addBegin = "<s> "
    addEnd = " </s>"
    
    with open(file_name, 'r') as f:
        file_lines = [''.join([addBegin, x.strip(), addEnd, '\n']) for x in f.readlines()]
    
    with open(file_name, 'w') as f:
        f.writelines(file_lines)
    
    file = open(file_name, 'r')
    
    output.write("The <s> and </s> tags are successfully padded to the file " + file_name + "\n")
    
    return file.read()

def resetTags(file_name, beginTag = "<s> ", endTag = " </s>"):
    """
    removes the tags added using addTags

    Args:
        file_name (string): a string of file path
        beginTag (string, optional): the begin tag. Defaults to "<s> ".
        endTag (string, optional): the end tag. Defaults to " </s>".
    """
    with open(file_name, 'r') as f:
        file_lines = [x.replace(beginTag, "") for x in f.readlines()]
        file_lines = [x.replace(endTag, "") for x in file_lines]
    with open(file_name, 'w')as f:
        f.writelines(file_lines)
    
    output.write("<s> and </s> tags are successfully removed from the file " + file_name + "\n")
    
train = addTags(train_path)
test = addTags(test_path)
sentence = addTags(sentence_path)

###  UNIGRAM MODEL ###

# create a unigram object
unigram = UnigramModel(train, test)

# store the dic of training and test corpora before padding unk tag
unigram_before_pad_train_dic = unigram.getTrainWord()
unigram_before_pad_test_dic = unigram.getTestWord()

# get the size of each dic
unigram_train_size = unigram.getTrainSize()
unigram_test_size = unigram.getTestSize()

# pad each dic with unk tag
unigram.padUnknownTraining(unigram_before_pad_train_dic)
unigram.padUnkownTest(unigram_before_pad_test_dic)

# get the dic after padding
unigram_after_pad_train_dic = unigram.getTrainWord()
unigram_after_pad_test_dic = unigram.getTestWord()

# get the dic of sentence after padding
unigram_sentence_dic = unigram.padUnknownSentence(sentence)

### debuggin section ###
# print("################################")
# print("unigram_before_pad_train_dic: ", unigram_before_pad_train_dic)
# print("unigram_before_pad_test_dic: ", unigram_before_pad_test_dic)
# print("unigram_train_size: ", unigram_train_size)
# print("unigram_test_size: ", unigram_test_size)
# print("unigram_after_pad_train_dic: ", unigram_after_pad_train_dic)
# print("unigram_after_pad_test_dic: ", unigram_after_pad_test_dic)
# print("unigram_sentence_dic: ", unigram_sentence_dic)
# print("################################")

######################

### Bigram Model ###
# create a bigram model
bigram = BigramModel()

# get the word arrays of all the corpora in order to proceed
bigram_training_words = bigram.padUnknown(train, unigram_after_pad_train_dic)
bigram_test_words = bigram.padUnknown(test, unigram_before_pad_train_dic)

# set dictionaries for training and test
bigram.setDictionary(bigram_training_words, True, False)
bigram.setDictionary(bigram_test_words,  False, True)

# get the dic of training and test
bigram_training_dic = bigram.getTrainWord()
bigram_test_dic = bigram.getTestWord()

bigram_sentence_words = bigram.padUnknown(sentence, unigram_before_pad_train_dic)
bigram_sentence_dic = bigram.setDictionary(bigram_sentence_words, False, False)

bigram_train_size = bigram.getTrainSize()
bigram_test_size = bigram.getTestSize()

### debuggin section ###
# print("################################")
# print("bigram_training_words: ", bigram_training_words)
# print("bigram_test_words: ", bigram_test_words)
# print("bigram_training_dic: ", bigram_training_dic)
# print("bigram_test_dic: ", bigram_test_dic)
# print("bigram_train_size: ", bigram_train_size)
# print("bigram_test_size: ", bigram_test_size)
# print("################################")

####################

### Bigram Add One Smoothing Model ###
# create a add one smoothing bigram model
smoothing = AddOneSmoothingBigram()

smoothing_training_words = smoothing.padUnknown(train, unigram_after_pad_train_dic)
smoothing_test_words = smoothing.padUnknown(test, unigram_before_pad_train_dic)

# set dictionaries for training and test
smoothing.setDictionary(smoothing_training_words, True, False)
smoothing.setDictionary(smoothing_test_words,  False, True)

# get the dic of training and test
smoothing_training_dic = smoothing.getTrainWord()
smoothing_test_dic = smoothing.getTestWord()


smoothing_sentence_words = smoothing.padUnknown(sentence, unigram_before_pad_train_dic)
smoothing_sentence_dic = smoothing.setDictionary(smoothing_sentence_words, False, False)

### debuggin section ###
# print("################################")
# print("smoothing_training_words: ", smoothing_training_words)
# print("smoothing_test_words: ", smoothing_test_words)
# print("smoothing_training_dic: ", smoothing_training_dic)
# print("smoothing_test_dic: ", smoothing_test_dic)
# print("################################")

######################################

# --------------for question 3 ------------------
type_count = 0
token_count = 0
for test_word in unigram_before_pad_test_dic:
    if test_word != "<s>" and test_word not in unigram_before_pad_train_dic:
        type_count += 1
        token_count += unigram_before_pad_test_dic[test_word]
unigram_word_type_not_occur_percentage = type_count / (len(unigram_before_pad_test_dic) - 1) * 100
unigram_word_token_not_occur_percentage = token_count / (unigram_test_size - unigram_before_pad_test_dic["<s>"]) * 100
# --------------for question 3 ------------------

# --------------for question 4 ------------------
type_count = 0
token_count = 0
start_count = 0
for test_word in bigram_test_dic:
    if "<s>" not in test_word:
        if test_word not in bigram_training_dic:
            type_count += 1
            token_count += bigram_test_dic[test_word]
    else:
        start_count +=1
bigram_word_type_not_occur_percentage = type_count / (len(bigram_test_dic) - 1) * 100
bigram_word_token_not_occur_percentage = token_count / (bigram_test_size - start_count) * 100
# --------------for question 4 ------------------

output.write("\n")

output.write("Q1.3.1\n")
output.write("How many word types (unique words) are there in the training corpus? Please include\n" 
           + "the end-of-sentence padding symbol </s> and the unknown token <unk>. Do not include\n"
           + "the start of sentence padding symbol <s>.\n")
output.write("The answer is: " + str(len(unigram_after_pad_train_dic) - 1) + "\n")

output.write("\n")

output.write("Q1.3.2\n")
output.write("How many word tokens are there in the training corpus? Do not include the start of\n"
           + "sentence padding symbol <s>.\n")
output.write("The answer is: " + str(unigram_train_size - unigram_before_pad_train_dic["<s>"]) + "\n")

output.write("\n")

output.write("Q1.3.3\n")
output.write("What percentage of word tokens and word types in the test corpus did not occur in\n"
           + "training (before you mapped the unknown words to <unk> in training and test data)?\n"
           + "Please include the padding symbol </s> in your calculations. Do not include the\n"
           + "start of sentence padding symbol <s>.\n")
output.write("The answer is: \n")
output.write("The percentage of word types in test not occur in train is: " + str(unigram_word_type_not_occur_percentage) + "%\n")
output.write("The percentage of word tokens in test not occur in train is: " + str(unigram_word_token_not_occur_percentage) + "%\n")

output.write("\n")

output.write("Q1.3.4\n")
output.write("Now replace singletons in the training data with <unk> symbol and map words (in the\n"
           + "test corpus) not observed in training to <unk>. What percentage of bigrams (bigram\n"
           + "types and bigram tokens) in the test corpus did not occur in training (treat <unk> as\n"
           + "a regular token that has been observed). Please include the padding symbol </s> in\n"
           + "your calculations. Do not include the start of sentence padding symbol <s>.\n")
output.write("The answer is: \n")
output.write("The percentage of bigram word types in test not occur in train is: " + str(bigram_word_type_not_occur_percentage) + "%\n")
output.write("The percentage of bigram word tokens in test not occur in train is: " + str(bigram_word_token_not_occur_percentage) + "%\n")

output.write("\n")

output.write("Q1.3.5\n")
output.write("Compute the log probability of the following sentence under the three models (ignore\n"
           + "capitalization and pad each sentence as described above). Please list all of the\n"
           + "parameters required to compute the probabilities and show the complete calculation.\n"
           + "Which of the parameters have zero values under each model? Use log base 2 in your\n"
           + "calculations. Map words not observed in the training corpus to the <unk> token.\n")
output.write("The answer is:\n")
### Unigram ###
output.write("For unigram model,\n")
unigram_sentence_log_prob = unigram.calculateLogProb(unigram_sentence_dic, output)
output.write("The log probability of the sentence under unigram model is: " + str(unigram_sentence_log_prob) + "\n")


### Bigram ###
output.write("For bigram modlel,\n")
bigram_sentence_log_prob = bigram.calculateLogProb(bigram_sentence_dic, unigram_after_pad_train_dic, output)
output.write("The log probability of the sentence under bigram model is: " + str(bigram_sentence_log_prob) + "\n")



### Add One Smoothing Bigram ###
output.write("For add-one-smoothing bigram model,\n")

smoothing_sentence_log_prob = smoothing.calculateLogProb(smoothing_sentence_dic, unigram_after_pad_train_dic, output)
output.write("The log probability of the sentence under add-one-smoothing bigram model is: " + str(smoothing_sentence_log_prob) + "\n")

output.write("\n")

output.write("Q1.3.6\n")
output.write("Compute the perplexity of the sentence above under each of the models.\n")

output.write("The answer is:\n")

### Unigram ###
output.write("For unigram model,\n")
unigram_sentence_perplexity = unigram.calculatePerplexity(unigram_sentence_dic)
output.write("The perplexity of the sentence under unigram model is: " + str(unigram_sentence_perplexity) + "\n")


### Bigram ###
output.write("For bigram modlel,\n")
bigram_sentence_perplexity = bigram.calculatePerplexity(bigram_sentence_dic, unigram_after_pad_train_dic)
output.write("The perplexity of the sentence under bigram model is: " + str(bigram_sentence_perplexity) + "\n")



### Add One Smoothing Bigram ###
output.write("For add-one-smoothing bigram model,\n")
smoothing_sentence_perplexity = smoothing.calculatePerplexity(smoothing_sentence_dic, unigram_after_pad_train_dic)
output.write("The perplexity of the sentence under add-one-smoothing bigram model is: " + str(smoothing_sentence_perplexity) + "\n")

output.write("\n")

output.write("Q1.3.7\n")
output.write("Compute the perplexity of the entire test corpus under each of the models. Discuss\n"
           + "the differences in the results you obtained.\n")

output.write("The answer is:\n")

### Unigram ###
output.write("For unigram model,\n")
unigram_test_perplexity = unigram.calculatePerplexity(unigram_after_pad_test_dic)
output.write("The perplexity of the test corpus under unigram model is: " + str(unigram_test_perplexity) + "\n")


### Bigram ###
output.write("For bigram modlel,\n")
bigram_test_perplexity = bigram.calculatePerplexity(bigram_test_dic, unigram_after_pad_train_dic)
output.write("The perplexity of the test corpus under bigram model is: " + str(bigram_test_perplexity) + "\n")



### Add One Smoothing Bigram ###
output.write("For add-one-smoothing bigram model,\n")
smoothing_test_perplexity = smoothing.calculatePerplexity(smoothing_test_dic, unigram_after_pad_train_dic)
output.write(" The perplexity of the test corpus under add-one-smoothing bigram model is: " + str(smoothing_test_perplexity) + "\n")


resetTags(train_path)
resetTags(test_path)
resetTags(sentence_path)
