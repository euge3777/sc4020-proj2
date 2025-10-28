import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureTransformer:
    def __init__(self, method='quantile', n_bins=3):
        """
        Initialize feature transformer
        
        Parameters:
        method: 'quantile', 'uniform', or 'kmeans'
        n_bins: number of categories (typically 3 for low/medium/high)
        """
        self.method = method
        self.n_bins = n_bins
        self.transformers = {}
        self.bin_labels = ['low', 'medium', 'high']
        
    def fit_transform(self, df, feature_columns):
        """
        Fit transformers and transform features
        """
        transformed_data = df.copy()
        
        for column in feature_columns:
            if column in df.columns:
                transformed_data[f"{column}_binned"] = self._transform_feature(
                    df[column], column
                )
        
        return transformed_data
    
    def _transform_feature(self, series, feature_name):
        """Transform a single feature based on the selected method"""
        if self.method == 'quantile':
            return self._quantile_binning(series, feature_name)
        elif self.method == 'uniform':
            return self._uniform_binning(series, feature_name)
        elif self.method == 'kmeans':
            return self._kmeans_binning(series, feature_name)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _quantile_binning(self, series, feature_name):
        """Quantile-based binning"""
        quantiles = [0, 0.33, 0.67, 1.0]
        bins = series.quantile(quantiles).values
        bins = np.unique(bins)  # Remove duplicates
        
        if len(bins) < 4:  # Handle case where values are identical
            bins = np.linspace(series.min(), series.max(), 4)
        
        binned = pd.cut(series, bins=bins, labels=self.bin_labels[:len(bins)-1], 
                       include_lowest=True)
        self.transformers[feature_name] = {'bins': bins, 'method': 'quantile'}
        return binned
    
    def _uniform_binning(self, series, feature_name):
        """Uniform width binning"""
        bins = np.linspace(series.min(), series.max(), self.n_bins + 1)
        binned = pd.cut(series, bins=bins, labels=self.bin_labels, 
                       include_lowest=True)
        self.transformers[feature_name] = {'bins': bins, 'method': 'uniform'}
        return binned
    
    def _kmeans_binning(self, series, feature_name):
        """K-means clustering based binning"""
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        
        # Standardize for k-means
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit k-means
        kmeans = KMeans(n_clusters=self.n_bins, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Sort clusters by their centers to create ordered categories
        centers = scaler.inverse_transform(kmeans.cluster_centers_).flatten()
        sorted_indices = np.argsort(centers)
        
        # Map cluster labels to low/medium/high based on sorted centers
        label_mapping = {sorted_indices[i]: self.bin_labels[i] 
                        for i in range(len(sorted_indices))}
        
        binned = pd.Series([label_mapping[label] for label in cluster_labels], 
                          index=series.index)
        
        self.transformers[feature_name] = {
            'kmeans': kmeans, 'scaler': scaler, 
            'label_mapping': label_mapping, 'method': 'kmeans'
        }
        return binned

class SequenceGenerator:
    def __init__(self, feature_importance_method='clinical'):
        """
        Generate sequences based on feature importance
        
        Parameters:
        feature_importance_method: 'clinical', 'statistical', or 'temporal'
        """
        self.feature_importance_method = feature_importance_method
        
    def generate_sequences(self, df, patient_id_col, diagnosis_col, 
                          transformed_features, max_length=5):
        """Generate sequences for each patient"""
        sequences = []
        
        for idx, row in df.iterrows():
            patient_id = row[patient_id_col]
            diagnosis = row[diagnosis_col]
            
            # Extract non-zero/non-low features for sequence
            feature_sequence = []
            for feature in transformed_features:
                binned_col = f"{feature}_binned"
                if binned_col in df.columns:
                    value = row[binned_col]
                    if value in ['medium', 'high']:  # Focus on elevated features
                        feature_sequence.append(f"{value}_{feature}")
            
            # Sort by clinical importance or statistical significance
            if self.feature_importance_method == 'clinical':
                feature_sequence = self._sort_by_clinical_importance(feature_sequence)
            
            # Limit sequence length
            feature_sequence = feature_sequence[:max_length]
            
            sequences.append({
                'patient_id': patient_id,
                'diagnosis': diagnosis,
                'sequence': ' '.join(feature_sequence),
                'sequence_length': len(feature_sequence)
            })
        
        return pd.DataFrame(sequences)
    
    def _sort_by_clinical_importance(self, feature_sequence):
        """Sort features by clinical importance for cancer diagnosis"""
        # Define clinical importance hierarchy
        importance_order = {
            'radius': 1, 'perimeter': 2, 'area': 3,
            'concave_points': 4, 'concavity': 5, 'compactness': 6,
            'texture': 7, 'smoothness': 8, 'symmetry': 9, 'fractal_dimension': 10
        }
        
        def get_importance(feature_name):
            for key in importance_order:
                if key in feature_name:
                    return importance_order[key]
            return 999  # Unknown features go to end
        
        return sorted(feature_sequence, key=get_importance)

# Usage example
def process_cancer_data():
    # Load original data
    df = pd.read_csv('data/Cancer_Data.csv')
    
    # Define feature columns (excluding id and diagnosis)
    feature_columns = [col for col in df.columns 
                      if col not in ['id', 'diagnosis']]
    
    # Process with different methods
    methods = ['quantile', 'uniform', 'kmeans']
    
    for method in methods:
        print(f"\n=== Processing with {method} method ===")
        
        # Transform features
        transformer = FeatureTransformer(method=method)
        transformed_df = transformer.fit_transform(df, feature_columns)
        
        # Generate sequences
        sequence_generator = SequenceGenerator()
        sequences_df = sequence_generator.generate_sequences(
            transformed_df, 'id', 'diagnosis', feature_columns
        )
        
        # Save results
        output_file = f'data/Cancer_Data_sequences_{method}.csv'
        sequences_df.to_csv(output_file, index=False)
        
        # Print statistics
        print(f"Generated {len(sequences_df)} sequences")
        print(f"Average sequence length: {sequences_df['sequence_length'].mean():.2f}")
        print(f"Malignant cases: {(sequences_df['diagnosis'] == 'M').sum()}")
        print(f"Benign cases: {(sequences_df['diagnosis'] == 'B').sum()}")

if __name__ == "__main__":
    process_cancer_data()