import numpy as np
from keras.preprocessing.sequence import pad_sequences

def idx_to_word(integer,tokenizer):    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model, feature,tokenizer, max_length):    
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        y_pred = model.predict([feature,sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
            
        in_text+= " " + word
        
        if word == 'endseq':
            break
    
    in_text = in_text.replace("startseq", "").replace("endseq", "")
    return in_text 