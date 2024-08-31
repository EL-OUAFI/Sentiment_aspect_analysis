# Sentiment_aspect_analysis

**Model 1: BERT-based Classifier**
    1.    Model: Pre-trained bert-base-uncased from Hugging Face.
    2.    Input Representation:
        Sentence: The input sentence containing the aspect term.
        Aspect Term: Marked with special tokens [T1] and [T2] around the term.
        Aspect Category: Mapped to an embedding specific to the category.
        Sentiment Score: Calculated using a sentiment lexicon tailored for each aspect.
    3.    Architecture:
        BERT Embedding: The sentence with marked aspect terms is passed through BERT to obtain contextual embeddings.
        Aspect Embedding: Each aspect category is mapped to a dense vector representation using an embedding layer.
        Sentiment Score: A sentiment score is computed based on a lexicon and passed through a small fully connected layer.
        Concatenation: The outputs from BERT, aspect embeddings, and sentiment embeddings are concatenated.
        Classification: The concatenated vector is passed through a fully connected layer with a softmax activation to predict the sentiment polarity (positive, negative, neutral).
        Loss Function: A hybrid loss function, combining Focal Loss and Label Smoothing, is used to handle class imbalance.




**Model 2: RoBERTa-LSTM Classifier**
    1.    Model: Pre-trained roberta-base from Hugging Face, combined with an LSTM layer.
    2.    Input Representation:
        Sentence: The input sentence with aspect terms marked similarly to the BERT model.
        Aspect Term: Encoded in the sentence with [T1] and [T2].
        Aspect Category: Mapped to an embedding.
        Sentiment Score: Incorporated from a lexicon-based sentiment analysis.
    3.    Architecture:
    RoBERTa Embedding: The sentence is passed through RoBERTa to produce contextual embeddings.
    LSTM Layer: The embeddings are processed through an LSTM to capture sequential dependencies.
    Aspect Embedding: Similar to the BERT model, the aspect is embedded and combined with the output of the LSTM.
    Sentiment Score: Processed through a fully connected layer before concatenation.
    Concatenation: The LSTM output, aspect embedding, and sentiment score are combined.
    Classification: The combined features are passed through a final fully connected layer to predict the sentiment polarity.
    Loss Function: A hybrid loss function, combining Focal Loss and Label Smoothing, is used to manage class imbalance.
***Nota :Data augmentation was implemented in this modelseparatelyusing paraphrasing, but it was not used in the final ensemble model due to increased complexity without significant accuracy improvement.***





**Ensemble Approach for both models**
    Motivation: After training and evaluating the BERT and RoBERTa-LSTM models individually (both achieving around 88% accuracy), the ensemble method was employed to leverage the strengths of both models.
    Method: The logits (output before softmax) from both models are averaged. This averaged logit is then passed through a softmax function to obtain the final predicted sentiment label.
    Outcome: The ensemble method improved the accuracy to 89.39% on the development dataset.

