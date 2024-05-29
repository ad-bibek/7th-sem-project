import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Additive smoothing parameter
        self.classes = None  # List of class labels
        self.class_probs = None  # Prior probabilities of each class
        self.word_probs = None  # Conditional probabilities of each word given each class
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_words = X.shape[1]
        self.class_probs = np.zeros(num_classes)
        self.word_probs = np.zeros((num_classes, num_words))
        
        # Calculate prior probabilities of each class
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_probs[i] = (len(X_c) + self.alpha) / (len(X) + self.alpha * num_classes)
            
            # Calculate conditional probabilities of each word given each class
            total_words = np.sum(X_c)
            for j in range(num_words):
                self.word_probs[i][j] = (np.sum(X_c[:, j]) + self.alpha) / (total_words + self.alpha * num_words)
    
    def predict(self, X):
        num_samples = X.shape[0]
        y_pred = np.zeros(num_samples)
        for i in range(num_samples):
            probs = np.zeros(len(self.classes))
            for j, c in enumerate(self.classes):
                # Calculate the log-likelihood of the sample belonging to class c
                log_likelihood = np.sum(np.log(self.word_probs[j]) * X[i]) + np.log(self.class_probs[j])
                probs[j] = log_likelihood
                
            # Use the maximum log-likelihood to predict the class of the sample
            y_pred[i] = self.classes[np.argmax(probs)]
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
