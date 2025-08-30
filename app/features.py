from typing import List, Tuple
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class CombinedTfidf:
    def __init__(self):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=2, max_df=0.9)
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_df=0.9)

    def fit_transform(self, texts: List[str]):
        Xw = self.word_vec.fit_transform(texts)
        Xc = self.char_vec.fit_transform(texts)
        return hstack([Xw, Xc])

    def transform(self, texts: List[str]):
        Xw = self.word_vec.transform(texts)
        Xc = self.char_vec.transform(texts)
        return hstack([Xw, Xc])

    def get_feature_names_out(self) -> np.ndarray:
        wnames = self.word_vec.get_feature_names_out()
        cnames = self.char_vec.get_feature_names_out()
        return np.concatenate([wnames, cnames])


def vectorize_text(train_texts: List[str], test_texts: List[str]) -> Tuple[CombinedTfidf, np.ndarray, np.ndarray]:
    vec = CombinedTfidf()
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return vec, X_train, X_test
