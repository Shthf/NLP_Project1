import math


class UnigramModel:
    def __init__(self, train_corpus, test_corpus):
        """Constructor of UnigramModel class that initialize everything
        

        Args:
            train_corpus (string): a string of the training corpus file (after padded)
            test_corpus (string): a string of the test text file(after padded)
        """
        
        self.train_word = {}
        self.train_total = 0
        self.test_word = {}
        self.test_total = 0
        
        self.setDictionary(train_corpus)
        self.setDictionary(test_corpus, False)
                
    def setDictionary(self, text, isTraining = True):
        """This method sets the given text to its corresponding dictionaries

        Args:
            text (string): a string of text (after padded)
            isTraining (bool, optional): indicates if the text if for training or test. Defaulted to true(training)
        """
        
        word_dic = {}
        total_count = 0
        
        for word in text.lower().split():
            total_count += 1
            if word in word_dic:
                word_dic[word] += 1
            else:
                word_dic[word] = 1
                
        if isTraining:
            self.train_word = word_dic.copy()
            self.train_total = total_count
        else:
            self.test_word = word_dic.copy()
            self.test_total = total_count
            
        del word_dic
    
    def getTrainWord(self):
        return self.train_word
    
    def getTrainSize(self):
        return self.train_total
    
    def getTestWord(self):
        return self.test_word  
    
    def getTestSize(self):
        return self.test_total  
            
    def padUnknownTraining(self, dic):
        """This method pads <unk> tag to all the words that only appear once in the text
           Only for training corpus

        Args:
            dic (dictionary): a dictionary of all words appeared in a text
        """
        
        count = 0
        temp = dic.copy()
        for word in self.train_word:
            if self.train_word[word] == 1:
                count += 1
                del temp[word]

        temp["<unk>"] = count
        
        self.train_word = temp
        del temp
        
    def padUnkownTest(self, dic):
        """padding unknown tag <unk> to each word in test corpus not showing in training

        Args:
            dic (dictionary): a dictionary of test corpus
        """
        count = 0
        temp_test = self.test_word.copy()
        for word in dic:
            if word not in self.train_word:
                count += 1
                del temp_test[word]
                
        temp_test["<unk>"] = count
        
        self.test_word = temp_test
        del temp_test
        
    def padUnknownSentence(self, sentence):
        """padding unknown tag <unk> to each word in sentence not showing in training and store the words into a dictionary

        Args:
            sentence (string): a string of sentence(input)
        """
        
        sentence_dic = {}
        for word in sentence.lower().split():
            if word in sentence_dic:
                sentence_dic[word] += 1
            else:
                sentence_dic[word] = 1
             
        count = 0
        for word in sentence_dic.copy():
            if word not in self.train_word:
                count += 1
                del sentence_dic[word]
        sentence_dic["<unk>"] = count
        
        return sentence_dic
                
    def calculateLogProb(self, dic, output, isSentence = True):
        """this method calculates the log probability of given sentence under unigram model

        Args:
            dic (string): a dictionary of words in test text or a sentence
            output (file): a file to write the output
            isSentence (boolen): whether if its calculating log prob for a sentence (defaulted to true)
        """
        
        log_prob = 0
        haveUndefined = False
        
        for word in dic:
            if word in self.train_word:
                log = math.log2(self.train_word[word] / self.train_total)
                
                if isSentence:
                    output.write("The log probability for the word (" + word + ") is " + str(log) + "\n")
                    
                log_prob += log
            else:
                haveUndefined = True
                if isSentence:
                    output.write("The log probability for the word (" + word + ") is undefined\n")
        
        if haveUndefined:
            return "undefined"
        return log_prob
    
    def calculatePerplexity(self, dic):
        """This method calculates the perplexity of a given dictionary using the formula and calculateLogProb method above

        Args:
            dic (dictionary): a dictionary of words in a text
        """
        log_prob = self.calculateLogProb(dic, None, False)
        
        if log_prob == "undefined":
            return "undefined"
        
        l = (1 / len(dic)) * log_prob
        p = 2 ** (-l)
        
        return p
    
       
            
    def printInfo(self):
        """debugging method
        """
        
        print("train word:", self.train_word)
        print("train count:", self.train_total)
        print("test word:", self.test_word)
        print("test count:", self.test_total)
        
    
    
        
        