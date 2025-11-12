import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def create_model(self):
        """Create and configure the Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=-1
        )
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the model and evaluate performance"""
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1_score': f1_score(y_test, y_pred_test, average='weighted')
        }
        
        # Generate detailed classification report
        metrics['classification_report'] = classification_report(y_test, y_pred_test)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)
        
        print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return self.model, metrics
    
    def save_model(self, filename='crop_recommendation_model.pkl'):
        """Save the trained model"""
        if self.model is not None:
            joblib.dump(self.model, filename)
            print(f"Model saved as {filename}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filename='crop_recommendation_model.pkl'):
        """Load a trained model"""
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return self.model
        except FileNotFoundError:
            print(f"Model file {filename} not found.")
            return None
    
    def predict(self, X):
        """Make predictions with the trained model"""
        if self.model is not None:
            return self.model.predict(X)
        else:
            print("No trained model available. Train or load a model first.")
            return None
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is not None:
            return self.model.predict_proba(X)
        else:
            print("No trained model available. Train or load a model first.")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            print("No trained model available or model doesn't support feature importance.")
            return None
