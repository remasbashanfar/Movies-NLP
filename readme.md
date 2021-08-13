# Introduction

Movie Tropes are common situations in movies that are repeated throughout a genre. Because both Media and Artificial Intelligence play a huge role in the modern world, analyzing movies can give us interesting patterns. The purpose of this research is to create an algorithm that clusters movies based on their tropes and genre. 

# Objective
- Defining tropes more precisely and cataloging them to facilitate computational analysis
- Training a Neural Network Model that groups movies by cosine similarity then comparing this model to pre-existing Wikipedia models

# Methodology
- Categorized and gave sample tropes from Horror and Comedy Movie genres.
- Tested trope accuracy in movies by researching keywords and by reading sample scripts.
- Used Natural Language Processing (NLTK, Python) & Neural Network Methods (Word2Vec, Doc2Vec) to train a model using 5900 movie synopsis as a database then created an algorithm that computes movies with specific tropes.

# How The Code Works
The algorithm used the model Word2Vec to get words with similar vectors. For instance, in the model, some words that have a similar vector to "serial killer" are "murderer" and "victim."  This targets similar words and uses them for the purposes of finding similar movies. Next, used a list<dictionary> to list movies and the number of times these keywords come up in them. Finally, sorted the list by the movies most accurate to the trope.
  
# Results
- This cosine similarity method gave 86% accuracy as 86% of the computed movies were in fact similar and had accurate tropes.
- Compared to the model that was trained with 5900 movies, the Wikipedia Embeddings Model gave more accurate word similarities because of its huger dataset.

# Conclusion
- The results show that the Word2Vec Neural Network Model is a promising  method for finding sentence-level similarities in text
- Even though categorizing tropes will be impactful for future research, testing them manually was not effective because of the time and word synonyms constraints. 

# Future?
- Creating an algorithm that analyzes nouns and verbs in movie scripts, relating them to genders and gender stereotypes in movies. 
- Bringing the research to life by developing a website that provides movie recommendations based on similar movie's plots and tropes.

  
###### References
- “Tv Tropes.” TV Tropes, tvtropes.org/.
- Kar, Sudipta. Folksonomication: Predicting Tags for Movies from Plot Synopses Using Emotion Flow Encoded Neural Network, 2018
- García-Ortega, Rubén Héctor. StarTroper, a Film Trope Rating Optimizer Using Machine Learning and Evolutionary Algorithms, 2020.

###### Acknowledgements
  I'd like to thank Professor Arnav Jhala and Mandar Chaudhary for their guidance throughout this research.
