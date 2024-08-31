from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer, AdamW, pipeline
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import random

#  Focal Loss for handling class imbalance.
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        
        """
        Initializes the FocalLoss module.
        Args:
            alpha (float): Weighting factor for the loss.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        
        """
        Computes the focal loss between logits and targets.
        Args:
            logits (Tensor): Predicted logits.
            targets (Tensor): Ground truth labels.
        Returns:
            Tensor: The computed focal loss.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
# Hybrid loss combining Focal Loss and Label Smoothing Cross-Entropy   
class HybridLoss(nn.Module):
    
    def __init__(self, focal_alpha=1, focal_gamma=2, label_smoothing=0.1, focal_weight=0.5):
        """
        Initializes the HybridLoss module.
        Args:
            focal_alpha (float): Weighting factor for the Focal Loss.
            focal_gamma (float): Focusing parameter for the Focal Loss.
            label_smoothing (float): Smoothing factor for Label Smoothing Cross-Entropy.
            focal_weight (float): Weighting factor to balance Focal Loss and Label Smoothing Cross-Entropy.
        """
        super(HybridLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        """
        Computes the hybrid loss by combining Focal Loss and Label Smoothing Cross-Entropy.
        Args:
            logits (Tensor): Predicted logits.
            targets (Tensor): Ground truth labels.
        Returns:
            Tensor: The computed hybrid loss.
        """
        focal_loss = self.focal_loss(logits, targets)
        label_smoothing_loss = self.label_smoothing_loss(logits, targets)
        loss = self.focal_weight * focal_loss + (1 - self.focal_weight) * label_smoothing_loss
        return loss

# Label Smoothing Cross-Entropy Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        Initializes the LabelSmoothingCrossEntropy module.
        Args:
            smoothing (float): Smoothing factor for label smoothing.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        """
        Computes the label smoothing cross-entropy loss.
        Args:
            logits (Tensor): Predicted logits.
            targets (Tensor): Ground truth labels.
        Returns:
            Tensor: The computed loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / log_probs.size(-1)
        loss = (-targets * log_probs).sum(dim=-1).mean()
        return loss

# Dataset for loading and processing data
class AspectSentimentDataset(Dataset):
    def __init__(self, data, tokenizer, aspect_mapping, label_encoder, sentiment_lexicon, max_length=128, augment=False):
        
        """
        Initializes the AspectSentimentDataset.
        Args:
            data (list): The dataset containing the sentences and labels.
            tokenizer (Tokenizer): Tokenizer for encoding the sentences.
            aspect_mapping (dict): Mapping of aspect categories to indices.
            label_encoder (LabelEncoder): Encoder for transforming labels to indices.
            sentiment_lexicon (dict): Lexicon of sentiment words and their corresponding scores.
            max_length (int): Maximum length for tokenized sequences.
            augment (bool): Whether to perform data augmentation.
        """
        
        self.data = data
        self.tokenizer = tokenizer
        self.aspect_mapping = aspect_mapping
        self.label_encoder = label_encoder
        self.sentiment_lexicon = sentiment_lexicon
        self.max_length = max_length
        self.augment = augment
        self.paraphraser = pipeline("text2text-generation", model="t5-small") if augment else None
        self.class_statistics = {
            "positive": 1057,
            "neutral": 58,
            "negative": 391
        }
        self.class_distribution = {
            "AMBIENCE#GENERAL": {"positive": 151, "neutral": 8, "negative": 29},
            "DRINKS#PRICES": {"positive": 8, "neutral": 1, "negative": 4},
            "DRINKS#QUALITY": {"positive": 36, "neutral": 0, "negative": 5},
            "DRINKS#STYLE_OPTIONS": {"positive": 24, "neutral": 0, "negative": 2},
            "FOOD#PRICES": {"positive": 29, "neutral": 0, "negative": 29},
            "FOOD#QUALITY": {"positive": 444, "neutral": 24, "negative": 135},
            "FOOD#STYLE_OPTIONS": {"positive": 64, "neutral": 7, "negative": 27},
            "LOCATION#GENERAL": {"positive": 11, "neutral": 5, "negative": 0},
            "RESTAURANT#GENERAL": {"positive": 110, "neutral": 1, "negative": 27},
            "RESTAURANT#MISCELLANEOUS": {"positive": 28, "neutral": 2, "negative": 9},
            "RESTAURANT#PRICES": {"positive": 6, "neutral": 1, "negative": 13},
            "SERVICE#GENERAL": {"positive": 144, "neutral": 7, "negative": 112},
        }
        if augment:
                self.paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
        else:
            self.paraphraser = None


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        """
        Retrieves the item at the specified index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            dict: A dictionary containing tokenized input, labels, aspect IDs, sentiment scores, and other information.
        """
        
        
        row = self.data[idx].strip().split('\t')
        polarity, aspect, term, offsets, sentence = row
        
        label = self.label_encoder.transform([polarity])[0]
        aspect_id = self.aspect_mapping[aspect]
    
        start, end = map(int, offsets.split(':'))
        term = sentence[start:end]
        
        pre_term = sentence[:start]
        post_term = sentence[end:]
        marked_sentence = f"{pre_term} [T1] {term} [T2] {post_term}"
        
        if self.augment and self.class_counts[polarity] < 100 and random.random() < 0.5:
            marked_sentence = self.paraphrase_sentence(marked_sentence)

        
        encoded = self.tokenizer(
            marked_sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        item['aspect_ids'] = torch.tensor(aspect_id, dtype=torch.long)
        item['sentence']= sentence
        item['term']= term
        # Calculate sentiment score using the specific term context
        sentiment_score = self._calculate_sentiment_score(sentence, aspect, term)
        item['sentiment_score'] = torch.tensor(sentiment_score, dtype=torch.float)

        return item

    def _calculate_sentiment_score(self, sentence, aspect, term):
        
        """
        Calculates a sentiment score for the given sentence, aspect, and term.
        Args:
            sentence (str): The sentence containing the term.
            aspect (str): The aspect category of the term.
            term (str): The specific term for which the sentiment is calculated.
        Returns:
            float: The calculated sentiment score.
        """
        
        words = sentence.lower().split()
        sentiment_score = 0
        lexicon = self.sentiment_lexicon.get(aspect, {})
        for word in words:
            if word in lexicon:
                sentiment_score += lexicon[word]
        if term in lexicon:
            sentiment_score += lexicon[term]
        return sentiment_score
    def paraphrase_sentence(self, sentence):
        """
        Generates a paraphrase of the sentence if data augmentation is enabled.
        Args:
            sentence (str): The sentence to paraphrase.
        Returns:
            str: The paraphrased sentence.
        """
        if self.paraphraser:
                paraphrased = self.paraphraser(sentence, num_return_sequences=1, do_sample=True)[0]['generated_text']
                return paraphrased
        return sentence
    

    

# Classifier with RoBERTa, LSTM, Attention, and Sentiment Lexicon
class CombinedRoBERTaLSTMClassifier(nn.Module):
    def __init__(self, roberta_model_name: str, num_aspects: int, aspect_embedding_dim: int, lstm_hidden_dim: int, num_labels: int):
        """
        Initializes the CombinedRoBERTaLSTMClassifier.
        Args:
            roberta_model_name (str): The name of the pre-trained RoBERTa model.
            num_aspects (int): The number of aspect categories.
            aspect_embedding_dim (int): The dimensionality of the aspect embeddings.
            lstm_hidden_dim (int): The hidden dimensionality of the LSTM.
            num_labels (int): The number of output labels (e.g., positive, negative, neutral).
        """
        super(CombinedRoBERTaLSTMClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.lstm = nn.LSTM(input_size=self.roberta.config.hidden_size, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.aspect_embedding = nn.Embedding(num_aspects, aspect_embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc_sentiment = nn.Linear(1, 32)  # Layer for sentiment score
        self.classifier = nn.Linear(lstm_hidden_dim * 2 + aspect_embedding_dim +32, num_labels ) 

    def forward(self, input_ids, attention_mask, aspect_ids, sentiment_score): 
        """
        Forward pass through the model.
        Args:
            input_ids (Tensor): Input IDs for the tokens in the sentence.
            attention_mask (Tensor): Attention mask to distinguish between actual tokens and padding.
            aspect_ids (Tensor): IDs representing the aspect categories.
            sentiment_score (Tensor): Sentiment scores derived from the sentiment lexicon.
        Returns:
            Tensor: Logits for the classification task.
        """
        # RoBERTa output
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # LSTM layer
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = lstm_output[:, -1, :]  # Get the last output from the LSTM

        # Aspect Embedding
        aspect_embeds = self.aspect_embedding(aspect_ids)
        
        # Sentiment score
        sentiment_embeds = self.fc_sentiment(sentiment_score.unsqueeze(1))

        # Combine LSTM output, Aspect Embedding, and Sentiment Score
        combined_output = torch.cat((lstm_output, aspect_embeds, sentiment_embeds), dim=1)  #, sentiment_embeds
        combined_output = self.dropout(combined_output)

        logits = self.classifier(combined_output)
        return logits

# Classifier for Roberta-LSTM model.
class Classifier:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.aspect_mapping = {
            'AMBIENCE#GENERAL': 0,
            'DRINKS#PRICES': 1,
            'DRINKS#QUALITY': 2,
            'DRINKS#STYLE_OPTIONS': 3,
            'FOOD#PRICES': 4,
            'FOOD#QUALITY': 5,
            'FOOD#STYLE_OPTIONS': 6,
            'LOCATION#GENERAL': 7,
            'RESTAURANT#GENERAL': 8,
            'RESTAURANT#MISCELLANEOUS': 9,
            'RESTAURANT#PRICES': 10,
            'SERVICE#GENERAL': 11
        }
        self.num_aspects = len(self.aspect_mapping)
        self.aspect_embedding_dim = 64
        self.lstm_hidden_dim = 128
        
        # Sentiment lexicon for sentiment scoring
        self.sentiment_lexicon =  {
            "FOOD#QUALITY": {
                "delicious": 3, "tasty": 2, "bland": -2, "disgusting": -3,
                "excellent": 3, "bad": -3, "poor": -2, "fresh": 2, "stale": -2,
                "perfect": 3, "savory": 2, "flavorful": 3, "gross": -3, "yummy": 2,
                "tender": 2, "dry": -2, "burnt": -3, "greasy": -2, "soggy": -2,
                "juicy": 2, "crispy": 2, "outstanding": 3, "disappointing": -2,
                "amazing": 3, "average": -1, "spicy": 2, "rich": 2, "sweet": 2,
                "flavorless": -2, "overcooked": -2, "undercooked": -2,
            },
            "FOOD#STYLE_OPTIONS": {
                "variety": 2, "limited": -2, "diverse": 2, "choices": 2, "options": 2,
                "boring": -2, "creative": 2, "unique": 2, "standard": -1, "specials": 2,
                "interesting": 2, "innovative": 2, "traditional": 1, "authentic": 2,
            },
            "FOOD#PRICES": {
                "cheap": 2, "expensive": -2, "affordable": 2, "overpriced": -3,
                "reasonable": 2, "value": 2, "bargain": 2, "costly": -2, "fair": 2,
                "rip-off": -3, "worth": 2, "pricey": -2, "deal": 2,
            },
            "SERVICE#GENERAL": {
                "prompt": 2, "slow": -2, "friendly": 2, "rude": -3, "helpful": 2,
                "unhelpful": -2, "great": 2, "terrible": -3, "excellent": 3, "poor": -2,
                "attentive": 2, "unattentive": -2, "professional": 2, "incompetent": -3,
                "courteous": 2, "neglectful": -2, "impeccable": 3, "horrible": -3,
                "unprofessional": -2,
            },
            "DRINKS#QUALITY": {
                "refreshing": 2, "flat": -2, "stale": -2, "strong": 2, "weak": -2,
                "perfect": 3, "diluted": -2, "delicious": 3, "bland": -2, "tasty": 2,
                "bitter": -2, "watery": -2, "smooth": 2, "crisp": 2, "amazing": 3,
            },
            "DRINKS#STYLE_OPTIONS": {
                "variety": 2, "limited": -2, "diverse": 2, "options": 2, "choices": 2,
                "boring": -2, "creative": 2, "unique": 2, "standard": -1, "selection": 2,
                "extensive": 2,
            },
            "DRINKS#PRICES": {
                "cheap": 2, "expensive": -2, "affordable": 2, "overpriced": -3,
                "reasonable": 2, "value": 2, "bargain": 2, "costly": -2, "fair": 2,
                "rip-off": -3, "deal": 2,
            },
            "AMBIENCE#GENERAL": {
                "cozy": 2, "romantic": 2, "noisy": -2, "quiet": 2, "loud": -2,
                "crowded": -2, "spacious": 2, "comfortable": 2, "uncomfortable": -2,
                "beautiful": 3, "charming": 2, "elegant": 3, "intimate": 2, "vibrant": 2,
                "dull": -2, "welcoming": 2, "inviting": 2, "cramped": -2, "serene": 2,
                "atmosphere": 2, "decor": 2, "clean": 2, "chic": 2,
            },
            "LOCATION#GENERAL": {
                "convenient": 2, "inconvenient": -2, "accessible": 2, "remote": -2,
                "central": 2, "far": -2, "close": 2, "nearby": 2, "distant": -2,
                "perfect": 3, "ideal": 2, "isolated": -2, "view": 2, "scenic": 2,
            },
            "RESTAURANT#GENERAL": {
                "great": 3, "good": 2, "bad": -3, "terrible": -3, "average": -1,
                "excellent": 3, "horrible": -3, "amazing": 3, "awful": -3, "fantastic": 3,
                "disappointing": -2, "mediocre": -2, "outstanding": 3, "charming": 2,
                "gem": 2, "must-visit": 3,
            },
            "RESTAURANT#PRICES": {
                "cheap": 2, "expensive": -2, "affordable": 2, "overpriced": -3,
                "reasonable": 2, "value": 2, "bargain": 2, "costly": -2, "fair": 2,
                "rip-off": -3, "deal": 2,
            },
            "RESTAURANT#MISCELLANEOUS": {
                "clean": 2, "dirty": -2, "sanitary": 2, "unsanitary": -2, "hygienic": 2,
                "messy": -2, "well-maintained": 2, "shabby": -2, "new": 2, "old": -2,
                "modern": 2, "dated": -2, "eco-friendly": 2, "organic": 2,
            }
        }
        # Initialize the model
        self.model = CombinedRoBERTaLSTMClassifier(
            roberta_model_name='roberta-base',
            num_aspects=self.num_aspects,
            aspect_embedding_dim=self.aspect_embedding_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            num_labels=3
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["positive", "negative", "neutral"])
        
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the provided training dataset.
        Args:
            train_filename (str): Path to the training dataset file.
            dev_filename (str): Path to the development dataset file.
            device (torch.device): The device to run the model on (e.g., CPU or GPU).
        """
        self.model.to(device)
        with open(train_filename, 'r', encoding='utf-8') as f:
            train_data = f.readlines()
        with open(dev_filename, 'r', encoding='utf-8') as f:
            dev_data = f.readlines()
        
        train_dataset = AspectSentimentDataset(train_data, self.tokenizer, self.aspect_mapping, self.label_encoder, self.sentiment_lexicon, augment=False)
        dev_dataset = AspectSentimentDataset(dev_data, self.tokenizer, self.aspect_mapping, self.label_encoder, self.sentiment_lexicon, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=16)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        criterion = HybridLoss(focal_weight=0.7) 
        self.model.train()
        epochs = 3
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                aspect_ids = batch['aspect_ids'].to(device)
                labels = batch['labels'].to(device)
                sentiment_score = batch['sentiment_score'].to(device)
                
                optimizer.zero_grad()
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aspect_ids=aspect_ids,
                    sentiment_score=sentiment_score
                )
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            dev_accuracy = self.evaluate(dev_loader, device)
            print(f"Development Set Accuracy after Epoch {epoch + 1}: {dev_accuracy:.2f}%")
    
    
    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """
        Predicts class labels for the input instances in the specified file.
        Args:
            data_filename (str): Path to the dataset file containing instances to predict.
            device (torch.device): The device to run the model on (e.g., CPU or GPU).
        Returns:
            List[str]: The list of predicted labels.
        """
        with open(data_filename, 'r', encoding='utf-8') as f:
            data = f.readlines()
        
        dataset = AspectSentimentDataset(data, self.tokenizer, self.aspect_mapping, self.label_encoder, self.sentiment_lexicon)
        data_loader = DataLoader(dataset, batch_size=16)
        
        self.model.to(device)
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                aspect_ids = batch['aspect_ids'].to(device)
                sentiment_score = batch['sentiment_score'].to(device)
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aspect_ids=aspect_ids,
                    sentiment_score=sentiment_score
                )
                
                preds = torch.argmax(logits, dim=1)
                preds = self.label_encoder.inverse_transform(preds.cpu().numpy())
                predictions.extend(preds)
        
        return predictions


    def evaluate(self, data_loader, device):
        """
        Evaluates the model on the development dataset.
        Args:
            data_loader (DataLoader): DataLoader for the development dataset.
            device (torch.device): The device to run the model on (e.g., CPU or GPU).
        Returns:
            float: The accuracy of the model on the development dataset.
        """
        self.model.eval()
        correct = 0
        total = 0


        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                aspect_ids = batch['aspect_ids'].to(device)
                labels = batch['labels'].to(device)
                sentiment_score = batch['sentiment_score'].to(device)
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aspect_ids=aspect_ids,
                    sentiment_score=sentiment_score
                )
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
               

        accuracy = (correct / total) * 100
        self.model.train()

        return accuracy
