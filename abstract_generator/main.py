import pickle
from tensorflow.keras.models import load_model
from abstract_puller import pull_pubmed_abstracts
from abstract_intelligence import AbstractMachine

################################
# Setting some parameters

# Always-necessary parameters
input_text = ["In this study, we were interested in estimation of the ",
              "Throughout the years, a major question in causal inference has been ",
              "our results were significant for an alpha of 0.05 (p=0."]
abstract_word_limit = 200  # general word-type limit to the abstract generator (only meant as approximate)
creativity = 0.5  # allows the 'creativity' for the model. The lower the value, the less creative the text generator

# Parameters if creating new model
create_new_model = False  # Whether to create a new model (if False, loads the pre-built model
training_set_size = 5000  # Number of abstracts to train with
epoch = 10  # training epochs for the RNN
pubmed_terms = '(causal inference) AND (English[Language])'  # PubMed search text for training set
email = "enter.your.own.email@email.com"  # email is given to NCBI when sending requests


################################
# Running script!

if __name__ == '__main__':
    # If a new model, create a new model
    if create_new_model:
        # Pulling a training set of abstracts
        abstracts = pull_pubmed_abstracts(search_terms=pubmed_terms,
                                          n_abstracts=training_set_size,
                                          email_address=email)
        abs_intel = AbstractMachine(list_of_abstracts=abstracts)
        print("Training characters:", len(abs_intel.text))
        if len(abs_intel.text) < 1e6:
            print("It is recommended to give a larger number of characters to train with. At least 100k is "
                  "recommended. 1 million is even better though.")
        abs_intel.fit(epochs=epoch, batch_size=128*2)

        # Saving all model output!
        text_model = abs_intel.fit_model
        text_model.save("abstract_brain.h5")
        params = [abs_intel.max_length,
                  abs_intel.chars,
                  abs_intel.char_index,
                  abs_intel.index_char]
        picklefile = open("model_params", "wb")
        pickle.dump(params, picklefile)
        picklefile.close()

    # Otherwise use the previously fit model
    else:
        # Loading pickle of parameters
        picklefile = open('model_params', 'rb')
        params = pickle.load(picklefile)
        picklefile.close()

        abs_intel = AbstractMachine(list_of_abstracts=["", ""])  # blank training set for loading a model
        abs_intel.fit_model = load_model("abstract_brain.h5")
        abs_intel.max_length = params[0]
        abs_intel.chars = params[1]
        abs_intel.char_index = params[2]
        abs_intel.index_char = params[3]

    # After model is estimated, start generating some text
    if isinstance(input_text, list):
        for input_t in input_text:
            abstract = abs_intel.predict_text(start_text=input_t,
                                              creativity=0.4,
                                              max_characters=abstract_word_limit*7)
            print("Input text:", input_t)
            print("New text:  ", abstract)

    elif isinstance(input_text, str):
        print("Input text:", input_text)
        abstract = abs_intel.predict_text(start_text=input_text,
                                          creativity=0.4,
                                          max_characters=abstract_word_limit * 7)
        print("New text:  ", abstract)

    else:
        print("invalid input :(")
