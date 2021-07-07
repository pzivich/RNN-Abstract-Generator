# Artifically Intelligent PubMed Abstract Generator

Paul Zivich

--------------------------------------

A simple recurrent neural network implementation to generate random abstracts. A trained version of the model is 
already saved and available. It can be ran by using the existing `main.py` file.

## Requirements

Python 3.6+ with the following libraries
- random
- pickle
- numpy
- biopython (for PubMed queries)
- tensorflow (for the RNN)

--------------------------------------

## Existing Model

To run the existing model, run the `main.py` file. There are several parameters the user could change
- `input_text` : change the input text to give to the RNN as a basis. The model was trained using 40 characters, so I 
  would provide at least 40 characters
- `abstract_word_limit` : more of an approximation than a limit. Determines how long the generated abstract should be
- `creativity` : controls the randomness of the predicted characters. Lower is less random, higher is more random

--------------------------------------

## Creating a New Model

WARNING: when training a new model, we query PubMed. Try not to query PubMed too often (so they don't block your IP).
If you provide your email, they should email before they block your IP address (I think...)

--------------------------------------

To train a new model, set `create_new_model = True`. The following parameters are then used
- `training_set_size` : number of abstracts to collect to train the RNN. Make sure this is smaller than number returned 
  by the PubMed search terms (it should still work though if there are less articles)
- `epoch` : training parameter for the RNN. 10 seems to work well (at least for me)
- `pubmed_terms` : search terms to narrow the pubmed search
- `email` : shared with NCBI when querying the PubMed database (input your own email! there is only a placeholder)

Once a new model is estimated, two new files are generated: `abstract_brain.h5` and `model_params`. The `.h5` file
consists of the saved keras model (so subsequent runs don't require re-estimation of the model). The `model_params`
file contains some parameters to give back to the keras RNN (which have been safely stored by `pickle` for us).
