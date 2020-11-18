# Social Networks?
### A CSCI 3725 Project by Stephen Crawford, Gerard Goucher, and Sam Roussel

## Setup

If necessary, install [NLTK](https://www.nltk.org/install.html). You may also
need to download Punkt for sentence tokenization. To do so, open a python terminal:

    $ import nltk
    $ nltk.download('punkt')

## Processing
This classifier begins by parsing a csv file with lines of the form [actor], [line of dialogue].
These are further processed into a dictionary mapping speakers to lines, and subsequently
into lists of word stems, classes (these are just the speakers), and documents (line/speaker tuples).

## Training Data
The actual training data is formed from the lists generated during processing. 
Each sentence becomes a bag of words, where for each word in the corpus, a 1 is added to the bag
if it is contained in the sentence, and a 0 is added if it is not in the sentence.

## Neural Network
This data is then used to train a feed-forward neural network which uses basic gradient descent and a standard
sigmoid function. The code for actually training the network and using it to classify sentences was provided by
Professor Harmon.

## Terms

    Hidden Layers -- Layers of neurons that do not directly interact with the input or output layers. This allows for 
    more complex functions to process data.
    
    Feedforward -- We are moving information forward in our network. The path is: input -> hidden -> output.
    
    Epochs -- A single step of training.
    
    Backpropogation -- A method of calculating the change in weights where the error is computed between the output 
    values and the correct values. The error value is then sent back through the network in order to help readjust weights.
    
    Alpha -- When calculating changes in weights, we have to prevent them from changing too much per iteration. To do 
    this, we use a small value between 0 and 1 known as the alpha value. This value ensures we do not over correct one 
    way or the other when adjusting weights. The learning rate/alpha value aims to make our network converge on valuable 
    output without taking too much time. 
    
