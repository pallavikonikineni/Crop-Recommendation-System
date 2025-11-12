import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import os

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
        
    def generate_sample_data(self):
        """Generate sample crop recommendation data"""
        np.random.seed(42)
        
        # Define crop types and their typical requirements
        crops_data = {
            'rice': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'temp': (20, 27), 'humidity': (80, 95), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
            'maize': {'N': (80, 120), 'P': (40, 60), 'K': (20, 40), 'temp': (18, 27), 'humidity': (55, 75), 'ph': (5.8, 7.0), 'rainfall': (50, 100)},
            'chickpea': {'N': (40, 70), 'P': (60, 85), 'K': (80, 120), 'temp': (17, 24), 'humidity': (10, 50), 'ph': (6.0, 7.5), 'rainfall': (30, 50)},
            'kidneybeans': {'N': (20, 40), 'P': (60, 80), 'K': (20, 40), 'temp': (15, 25), 'humidity': (18, 65), 'ph': (5.5, 7.0), 'rainfall': (60, 90)},
            'pigeonpeas': {'N': (20, 40), 'P': (60, 80), 'K': (20, 40), 'temp': (18, 29), 'humidity': (50, 70), 'ph': (5.5, 7.5), 'rainfall': (60, 120)},
            'mothbeans': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 'temp': (24, 28), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (30, 50)},
            'mungbean': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 'temp': (25, 30), 'humidity': (75, 85), 'ph': (6.2, 7.2), 'rainfall': (50, 75)},
            'blackgram': {'N': (40, 60), 'P': (60, 80), 'K': (20, 40), 'temp': (25, 35), 'humidity': (65, 75), 'ph': (5.5, 7.0), 'rainfall': (60, 100)},
            'lentil': {'N': (20, 40), 'P': (60, 80), 'K': (20, 40), 'temp': (15, 25), 'humidity': (65, 85), 'ph': (6.0, 7.5), 'rainfall': (25, 50)},
            'pomegranate': {'N': (19, 30), 'P': (5, 15), 'K': (5, 15), 'temp': (18, 35), 'humidity': (35, 55), 'ph': (5.5, 7.2), 'rainfall': (50, 100)},
            'banana': {'N': (100, 120), 'P': (75, 85), 'K': (50, 60), 'temp': (26, 30), 'humidity': (75, 85), 'ph': (5.5, 7.0), 'rainfall': (100, 180)},
            'mango': {'N': (19, 25), 'P': (5, 15), 'K': (5, 15), 'temp': (24, 27), 'humidity': (50, 70), 'ph': (5.5, 7.5), 'rainfall': (90, 150)},
            'grapes': {'N': (23, 30), 'P': (5, 15), 'K': (5, 15), 'temp': (8, 22), 'humidity': (80, 90), 'ph': (5.5, 7.0), 'rainfall': (50, 100)},
            'watermelon': {'N': (100, 120), 'P': (40, 50), 'K': (50, 60), 'temp': (24, 27), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (40, 60)},
            'muskmelon': {'N': (100, 120), 'P': (40, 50), 'K': (50, 60), 'temp': (24, 27), 'humidity': (90, 95), 'ph': (6.0, 7.0), 'rainfall': (20, 40)},
            'apple': {'N': (20, 30), 'P': (125, 135), 'K': (200, 210), 'temp': (21, 24), 'humidity': (90, 95), 'ph': (5.5, 7.0), 'rainfall': (100, 180)},
            'orange': {'N': (20, 30), 'P': (10, 15), 'K': (10, 15), 'temp': (15, 27), 'humidity': (75, 85), 'ph': (6.0, 7.5), 'rainfall': (100, 120)},
            'papaya': {'N': (50, 60), 'P': (55, 65), 'K': (50, 60), 'temp': (25, 30), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (100, 180)},
            'coconut': {'N': (20, 30), 'P': (10, 20), 'K': (30, 40), 'temp': (27, 30), 'humidity': (70, 80), 'ph': (5.2, 8.0), 'rainfall': (150, 250)},
            'cotton': {'N': (120, 140), 'P': (40, 50), 'K': (40, 50), 'temp': (21, 30), 'humidity': (80, 90), 'ph': (5.8, 8.0), 'rainfall': (50, 100)},
            'jute': {'N': (80, 100), 'P': (40, 50), 'K': (40, 50), 'temp': (25, 35), 'humidity': (70, 80), 'ph': (6.0, 7.5), 'rainfall': (150, 250)},
            'coffee': {'N': (100, 120), 'P': (15, 25), 'K': (30, 40), 'temp': (23, 25), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (150, 250)}
        }
        
        # Generate data for each crop
        all_data = []
        samples_per_crop = 100
        
        for crop, ranges in crops_data.items():
            for _ in range(samples_per_crop):
                # Generate random values within specified ranges
                row = {
                    'N': np.random.uniform(ranges['N'][0], ranges['N'][1]),
                    'P': np.random.uniform(ranges['P'][0], ranges['P'][1]),
                    'K': np.random.uniform(ranges['K'][0], ranges['K'][1]),
                    'temperature': np.random.uniform(ranges['temp'][0], ranges['temp'][1]),
                    'humidity': np.random.uniform(ranges['humidity'][0], ranges['humidity'][1]),
                    'ph': np.random.uniform(ranges['ph'][0], ranges['ph'][1]),
                    'rainfall': np.random.uniform(ranges['rainfall'][0], ranges['rainfall'][1]),
                    'label': crop
                }
                all_data.append(row)
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(all_data)
        df = shuffle(df, random_state=42).reset_index(drop=True)
        
        return df
    
    def load_data(self):
        """Load data from CSV file or generate sample data"""
        try:
            # Try to load from CSV file first
            if os.path.exists('crop_data.csv'):
                data = pd.read_csv('crop_data.csv')
                print("Loaded data from crop_data.csv")
            else:
                # Generate sample data if CSV doesn't exist
                data = self.generate_sample_data()
                # Save generated data to CSV for future use
                data.to_csv('crop_data.csv', index=False)
                print("Generated sample data and saved to crop_data.csv")
            
            self.data = data
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Fallback to generating sample data
            data = self.generate_sample_data()
            self.data = data
            return data
    
    def preprocess_data(self, data):
        """Preprocess the data for machine learning"""
        # Separate features and target
        feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = data[feature_columns]
        y = data['label']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder
    
    def load_and_preprocess_data(self):
        """Complete data loading and preprocessing pipeline"""
        # Load data
        data = self.load_data()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, label_encoder = self.preprocess_data(data)
        
        return X_train, X_test, y_train, y_test, label_encoder
    
    def get_raw_data(self):
        """Get the raw data for visualization"""
        if self.data is None:
            self.load_data()
        return self.data
    
    def get_feature_statistics(self):
        """Get statistics for features"""
        if self.data is None:
            self.load_data()
        
        feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        return self.data[feature_columns].describe()
