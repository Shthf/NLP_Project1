import math


class BigramModel:
    def __init__(self):
        
        
        self.train_word = {}
        self.train_size = 0
        self.test_word = {}
        self.test_size = 0
        
    def setDictionary(self, words, isTraining, isTest):
        """This method sets the given text to its corresponding dictionaries

        Args:
            words (array): a array of words of text (after padded)
            isTraining (bool): indicates if the text if for training
            isTest (bool): indicates if the text is for test
        """
        
        dic = {}
    
        for i in range(len(words) - 1):
            prev_word = words[i]
            curr_word = words[i + 1]
            
            if (prev_word,curr_word) in dic:
                dic[(prev_word, curr_word)] += 1
            else:
                dic[(prev_word, curr_word)] = 1
        
        if isTraining and not isTest:
            self.train_word = dic.copy()
            self.train_size = sum(dic.values())
        elif not isTraining and isTest:
            self.test_word = dic.copy()
            self.test_size = len(dic.values())
        else:
            return dic 
    
    def getTrainWord(self):
        return self.train_word

    def getTestWord(self):
        return self.test_word
    
    def getTrainSize(self):
        return self.train_size
    
    def getTestSize(self):
        return self.test_size
        
    # This method can be used for tagging both the training corpus or the test corpus
    # if tagging training corpus, the parameters will be training text, and unigram training dic after tagging
    # if tagging test corpus, the parameters will be test text, and unigram training dic before tagging
    def padUnknown(self, text, unigram_training_dic):
        """pads the unknown tag <unk> to each word appearing once in the corpus
           returning an array

        Args:
            text (string): a string of the text
            unigram_training_dic (dictionary): a dictionary of words in the unigram training text
        """
        
        words = text.lower().split()
        
        for i in range(len(words)):
            if words[i] not in unigram_training_dic:
                words[i] = "<unk>"

        return words
    
    def calculateLogProb(self, dic, unigram_dic, output, isSentence = True):
        """this method calculates the log probability of given sentence under bigram model

        Args:
            dic (dictionary): a dictionary of words in test text or a sentence in a bigram
            unigram_dic (dictionary): a dictionary of words in the unigram model
            output (file): a file to write the output
            isSentence (boolen): whether if its calculating log prob for a sentence (defaulted to true)
        """
        
        log_prob = 0
        haveUndefined = False
        
        for word in dic:
            if word in self.train_word:
                log = math.log2(self.train_word[word] / unigram_dic[word[0]])
                
                if isSentence:
                    output.write("The log probability for the word (" + str(word) + ") is " + str(log) + "\n")
                
                log_prob += log
            else:
                haveUndefined = True
                if isSentence:
                    output.write("The log probability for the word (" + str(word) + ") is undefined\n")
        
        if haveUndefined:
            return "undefined" 
           
        return log_prob 
        
    def calculatePerplexity(self, dic, unigram_dic):
        """This method calculates the perplexity of a given dictionary using the formula and calculateLogProb method above

        Args:
            dic (dictionary): a dictionary of words in a text
            unigram_dic (dictionary): a dictionary of words in the unigram model
        """
        
        log_prob = self.calculateLogProb(dic, unigram_dic, None, False)
        
        
        if log_prob == "undefined":
            return "undefined"
        
        l = (1 / sum(unigram_dic.values())) * log_prob
        p = 2 ** (-l)
        
        return p
                            
        
    def printInfo(self):
        print("train_word: ", self.train_word)
        print("train_size: ", self.train_size)
        print("test_word: ", self.test_word)
        print("test_size: ", self.test_size)
        
        
class AddOneSmoothingBigram(BigramModel):
    
    
    def __init__(self):
        super()
        
        
    def calculateLogProb(self, dic, unigram_dic, output, isSentence = True):
        """this method calculates the log probability of given sentence under bigram add one smoothing model

        Args:
            dic (dictionary): a dictionary of words in test text or a sentence in a bigram
            unigram_dic (dictionary): a dictionary of words in the unigram model
            output (file): a file to write the output
            isSentence (boolen): whether if its calculating log prob for a sentence (defaulted to true)
        """
        
        log_prob = 0
        
        vocab = len(unigram_dic) # total number of words in the corpus
        
        for word in dic:
            if word in self.train_word:
                log = math.log2((self.train_word[word] + 1)/ (unigram_dic[word[0]] + vocab))
            else:
                log = math.log2(1 / (vocab))  
            if isSentence:
                output.write("The log probability for the word (" + str(word) + ") is " + str(log) + "\n")
                
            log_prob += log
        
        return log_prob 
        
        
    
    def calculatePerplexity(self, dic, unigram_dic):
        """This method calculates the perplexity of a given dictionary using the formula and calculateLogProb method above

        Args:
            dic (dictionary): a dictionary of words in a text
            unigram_dic (dictionary): a dictionary of words in the unigram model
        """
        
        log_prob = self.calculateLogProb(dic, unigram_dic, None, False)
        
        
        if log_prob == "undefined":
            return "undefined"
        
        l = (1 / sum(unigram_dic.values())) * log_prob
        p = 2 ** (-l)
        
        return p