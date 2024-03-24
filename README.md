# NLP_Project1 Language Modelling

# Instructions
  1.  Download all files into the same directory of your choice except "output.txt" because this is the answer to the project with add-ons (The program will automatically generate a new "output.txt" in the same directory for results
  2.  Launch Terminal
  3.  Navigate to the directory by using "cd"
  4.  Run main.py by typing "python3 main.py" in the terminal
  5.  Open "output.txt" for results

# Notes
-  "mini-train.txt", "mini_test.txt", and "mini_sentence.txt" are used to test the results with smaller corpora
-  There are some debugging sections in "main.py" that are commented out, they print out all the data in the model, you can uncomment and print them out in the terminal for validation

# Project Requirements
In this assignment, you will train several language models and will evaluate them on a test corpus. Two files are provided with this assignment:

1. train.txt
2. test.txt

Each file is a collection of texts, one sentence per line. train.txt contains about 100,000 sentences from the NewsCrawl corpus. You will use this corpus to train the language models. The test corpus test.txt is from the same domain and will be used to evaluate the language models that you trained.

1.1 PRE-PROCESSING
Prior to training, please complete the following pre-processing steps:
1. Pad each sentence in the training and test corpora with start and end symbols(you can use "\<s>\" and "\</s>\", respectively).
2. Lowercase all words in the training and test corpora. Note that the data already has been tokenized (i.e. the punctuation has been split off words).
3. Replace all words occurring in the training data once with the token "\<unk>\". Every word in the test data not seen in training should be treated as "\<unk>\".

1.2 TRAINING THE MODELS
Please use train.txt to train the following language models:
1. An unigram maximum likelihood model.
2. A bigram maximum likelihood model.
3. A bigram model with Add-One smoothing.

1.3 QUESTIONS
1. How many word types (unique words) are there in the training corpus? Please include the end-of-sentence padding symbol "\</s>\" and the unknown token "\<unk>\". Do not include the start of sentence padding symbol "\<s>\".
2. How many word tokens are there in the training corpus? Do not include the start of sentence padding symbol "\<s>\".
3. What percentage of word tokens and word types in the test corpus did not occur in training (before you mapped the unknown words to "\<unk>\" in training and test data)? Please include the padding symbol "\</s>\" in your calculations. Do not include the start of sentence padding symbol "\<s>\".
4. Now replace singletons in the training data with "\<unk>\" symbol and map words(in the test corpus) not observed in training to "\<unk>\". What percentage of bigrams (bigram types and bigram tokens) in the test corpus did not occur in training (treat "\<unk>\" as a regular token that has been observed). Please include the padding symbol "\</s>\" in your calculations. Do not include the start of sentence padding symbol "\<s>\".
5. Compute the log probability of the following sentence under the three models (ignore capitalization and pad each sentence as described above). Please list all  parameters required to compute the probabilities and show the complete calculation. Which of the parameters have zero values under each model? Use log base 2 in your calculations. Map words not observed in the training corpus to the "\<unk>\" token.
   I look forward to hearing your reply .
6. Compute the perplexity of the sentence above under each of the models.
7. Compute the perplexity of the entire test corpus under each of the models. Discuss the differences in the results you obtained.
















