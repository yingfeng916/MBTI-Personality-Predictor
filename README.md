MBTI personality PREDICTOR

Inspiration:

- Different personalities choose different topics to talk about and post differently on Social Media.
- Question: Is it possible to predict someone’s personality type without having to go through the long questionnaires and by just inputting some text or posts from their social Media?
- Answer: Neuro Linguistic Programming (NLP) is a collection of techniques that can help to identify how people think, how they communicate and how they behave. In other words, NLP can be used to detect patterns in people’s behaviour

Business Value:

Corporation:
Human Resource/Recruitment team could request text input / social Media  from applicants to predict their personality and  decide if they are a good fit for the team dynamic/job role
Marketing team could use this to more effectively target the specific type of personalities in their client base/target audience without them having to fill out anything

Educational Institutes (School, college, or university) :
Enhance their impact by equipping the students to make the right choices 
Provide a greater return on investment and better educational outcomes

Automated personality prediction from social media :
Social Media Marketing 
political campaigns targeted ads
Dating applications and websites 

Personality prediction from text instead of long questionnaires:
Human resource management
Career selection 
Educational path selection

Data used:
Machine Learning classifier to predict MBTI personality type of an individual based on text samples from their posts on a personality forum
Source: Kaggle
Type: CSV containing 8675 rows 
Composition: 2 columns 
1. Personality type
2. combination of 50 posts from a person taken from a personality forum

Machine learning Approach:
Preprocessing
Remove links , Punctuation and any junk we don't want to use in the post.
Tokenization
Stopword removal + selected word removal
hashing term frequency
Fit the IDF model
Feature Vectorization

Model:
Naivebayes based on Bayesian Model in Natural Language processing.

ML results:
If it was a blind guess , each of four classifiers would only predict our data by 50% accuracy, however we can now see that each of them are much better than that.
While this seems to indicate a weak overall ability of our model to correctly classify all four MBTI dimensions, The overall accuracy of our trained model is 25% which is way higher than normal probability of 1/16 (6%).

