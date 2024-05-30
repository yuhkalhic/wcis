import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

class SentiWordNetClassifier:
    def __init__(self):
        # 确保已经下载了所需的 NLTK 资源
        nltk.download('sentiwordnet')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')

        # 初始化词形还原器
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):
        """将POS tag从Penn Treebank转换为WordNet的标签格式."""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    def analyze_sentiment(self, text):
        """计算文本的情感得分并返回二分类标签."""
        sentiment = 0.0
        tokens_count = 0

        words = word_tokenize(text)
        tagged_words = pos_tag(words)

        for word, tag in tagged_words:
            wn_tag = self.get_wordnet_pos(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
                continue

            lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

        # 根据得分确定情感标签
        if tokens_count > 0:
            sentiment /= tokens_count

        return 1 if sentiment > 0 else 0

    def evaluate_model(self, model, test_loader, device):
        """Evaluate the model using the given test_loader."""
        y_true = []
        y_pred = []

        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                predictions = model(inputs.to(device))
                predicted_labels = torch.argmax(predictions, dim=1)
                y_true.extend(targets.tolist())
                y_pred.extend(predicted_labels.cpu().tolist())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }
