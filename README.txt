Folders:
    - flask-folder: It contains the consolidated prototype
    - python_files: 
        - lda.py: Implementation of the LDA algorithm
        - classify_title.py: Trains the Naive Bayes model on the output of the LDA
        - classify.py: Combination of the LDA and the prediction of Naive Bayes on a new post
        - bag_of_words.py: Picks only the words with top frequencies to be the tags
        - tfidf.py: Uses the tf-idf algorithm on the posts and picks the words with the heighest weights

        - model.sav: Contains the saved Naive Bayes model
        - output.txt: Output of the LDA

        - Articles.csv: The dataset used