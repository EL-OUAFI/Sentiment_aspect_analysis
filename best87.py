from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder

class AspectSentimentDataset(Dataset):
    
    """
        Initializes the dataset object.
        Args:
            data: The dataset as a list of strings.
            tokenizer: The tokenizer for BERT.
            aspect_mapping: A dictionary mapping aspect categories to numeric IDs.
            label_encoder: The encoder for sentiment labels.
            sentiment_lexicon: A dictionary mapping aspect terms to sentiment scores.
            max_length: The maximum length for tokenization.
    """
    
    
    def __init__(self, data, tokenizer, aspect_mapping, label_encoder, sentiment_lexicon, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.aspect_mapping = aspect_mapping
        self.label_encoder = label_encoder
        self.sentiment_lexicon = sentiment_lexicon
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        """
        Processes each data item and returns a dictionary of features.
        Args:
            idx: The index of the data item.
        Returns:
            A dictionary containing tokenized inputs, labels, and sentiment scores.
        """
        
        
        row = self.data[idx].strip().split('\t')
        polarity, aspect, term, offsets, sentence = row
        
        # Extract the term from the sentence using offsets
        
        label = self.label_encoder.transform([polarity])[0]
        aspect_id = self.aspect_mapping[aspect]

        
        start, end = map(int, offsets.split(':'))
        term = sentence[start:end]
        
        
        pre_term = sentence[:start]
        post_term = sentence[end:]
        
        # Create the marked sentence with term surrounded by special tokens
        
        marked_sentence = f"{pre_term} [T1] {term} [T2] {post_term}"
        
        encoded = self.tokenizer(
            marked_sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare the dictionary of features
        
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        item['aspect_ids'] = torch.tensor(aspect_id, dtype=torch.long)
        # Calculate sentiment score using the specific term context
        sentiment_score = self._calculate_sentiment_score(sentence, aspect, term)
        item['sentiment_score'] = torch.tensor(sentiment_score, dtype=torch.float)

        return item
    
    def _calculate_sentiment_score(self, sentence, aspect, term):
        
        """
        Calculates the sentiment score for the term in the context of the sentence.
        Args:
            sentence: The full sentence.
            aspect: The aspect category.
            term: The target term within the sentence.
        Returns:
            A sentiment score as a float.
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
    

# BERT-based model for aspect-based sentiment classification

class AspectBertClassifier(nn.Module):
    def __init__(self, bert_model_name: str, num_aspects: int, aspect_embedding_dim: int, num_labels: int):
        
        """
        Initializes the BERT-based classifier.
        Args:
            bert_model_name: The name of the BERT model to be used.
            num_aspects: The number of unique aspect categories.
            aspect_embedding_dim: The dimensionality of the aspect embeddings.
            num_labels: The number of output labels (positive, negative, neutral).
        """
        
        super(AspectBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.aspect_embedding = nn.Embedding(num_aspects, aspect_embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc_sentiment = nn.Linear(1, 32)  # Layer for sentiment score
        combined_size = self.bert.config.hidden_size + aspect_embedding_dim + 32
        self.classifier = nn.Linear(combined_size, num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids, aspect_ids, sentiment_score):
        
        """
        Forward pass for the model.
        Args:
            input_ids: The tokenized input IDs.
            attention_mask: The attention mask for the inputs.
            token_type_ids: The token type IDs (segment IDs) for the inputs.
            aspect_ids: The aspect category IDs.
            sentiment_score: The sentiment score from the lexicon.
        Returns:
            Logits for the sentiment classes.
        """
        
        
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = bert_outputs.pooler_output
        
        aspect_embeds = self.aspect_embedding(aspect_ids)
        sentiment_embeds = self.fc_sentiment(sentiment_score.unsqueeze(1))
        
        combined = torch.cat((cls_output, aspect_embeds, sentiment_embeds), dim=1)
        combined = self.dropout(combined)
        
        logits = self.classifier(combined)
        return logits
    
    
    
    
#  Classifier class

class Classifier:
    def __init__(self):
        
        """
        Initializes the Classifier class, setting up tokenizers, mappings, and the model.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
        
        
        # Sentiment lexicon for scoring
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
        

        self.model = AspectBertClassifier(
            bert_model_name='bert-base-uncased',
            num_aspects=self.num_aspects,
            aspect_embedding_dim=self.aspect_embedding_dim,
            num_labels=3
        )
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["positive", "negative", "neutral"])
        
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        
        
        
        self.model.to(device)
        with open(train_filename, 'r', encoding='utf-8') as f:
            train_data = f.readlines()
        with open(dev_filename, 'r', encoding='utf-8') as f:
            dev_data = f.readlines()
        
        train_dataset = AspectSentimentDataset(train_data, self.tokenizer, self.aspect_mapping, self.label_encoder, self.sentiment_lexicon)
        dev_dataset = AspectSentimentDataset(dev_data, self.tokenizer, self.aspect_mapping, self.label_encoder, self.sentiment_lexicon)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=16)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)  
        

    
        self.model.train()
        epochs = 2
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                aspect_ids = batch['aspect_ids'].to(device)
                labels = batch['labels'].to(device)
                sentiment_score = batch['sentiment_score'].to(device)
                
                optimizer.zero_grad()
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    aspect_ids=aspect_ids,
                    sentiment_score=sentiment_score
                )
                
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            dev_accuracy = self.evaluate(dev_loader, device)
            print(f"Development Set Accuracy after Epoch {epoch + 1}: {dev_accuracy:.2f}%")
            
            
            
    
    def predict(self, data_filename: str, device: torch.device) -> List[str]:
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
                token_type_ids = batch['token_type_ids'].to(device)
                aspect_ids = batch['aspect_ids'].to(device)
                sentiment_score = batch['sentiment_score'].to(device)
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    aspect_ids=aspect_ids,
                    sentiment_score=sentiment_score
                )
                
                preds = torch.argmax(logits, dim=1)
                preds = self.label_encoder.inverse_transform(preds.cpu().numpy())
                predictions.extend(preds)
        
        return predictions
    
    
    
    

    def evaluate(self, data_loader, device):
        
        """
        Evaluates the model's accuracy on a given dataset.
        Args:
            data_loader: The DataLoader for the dataset to evaluate.
            device: The device to use for evaluation (CPU or GPU).
        Returns:
            The accuracy as a percentage.
        """
        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                aspect_ids = batch['aspect_ids'].to(device)
                labels = batch['labels'].to(device)
                sentiment_score = batch['sentiment_score'].to(device)
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    aspect_ids=aspect_ids,
                    sentiment_score=sentiment_score
                )
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = (correct / total) * 100
        self.model.train()
        return accuracy
    
    
    
