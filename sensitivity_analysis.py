import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class SensitivityAnalyzer:
    def __init__(self):
        self.results = {}
    
    def analyze_binning_methods(self):
        """Compare performance across different binning methods"""
        methods = ['quantile', 'uniform', 'kmeans']
        
        for method in methods:
            try:
                # Load sequence data
                df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
                
                # Analyze sequence patterns
                self.results[method] = self._analyze_patterns(df, method)
            except FileNotFoundError:
                print(f"Warning: File 'data/Cancer_Data_sequences_{method}.csv' not found.")
                continue
        
        return self.results
    
    def _analyze_patterns(self, df, method_name):
        """Analyze patterns in sequence data"""
        results = {
            'method': method_name,
            'total_sequences': len(df),
            'avg_sequence_length': df['sequence_length'].mean(),
            'malignant_patterns': self._extract_common_patterns(
                df[df['diagnosis'] == 'Malignant']
            ),
            'benign_patterns': self._extract_common_patterns(
                df[df['diagnosis'] == 'Benign']
            )
        }
        
        return results
    
    def _extract_common_patterns(self, df):
        """Extract most common patterns from sequences"""
        # Count individual features
        feature_counts = {}
        for sequence in df['sequence']:
            if pd.isna(sequence):  # Handle empty sequences
                continue
            features = sequence.split()
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Return top 10 most common features
        sorted_features = sorted(feature_counts.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_features[:10]
    
    def validate_biological_relevance(self):
        """Validate that patterns align with known cancer biology"""
        validation_results = {}
        
        # Known important cancer biomarkers
        important_features = [
            'radius', 'area', 'perimeter',  # Size-related
            'concave_points', 'concavity',  # Shape irregularity
            'compactness', 'texture'        # Tissue characteristics
        ]
        
        for method in ['quantile', 'uniform', 'kmeans']:
            try:
                df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
                malignant_df = df[df['diagnosis'] == 'Malignant']
                
                # Check if important features appear frequently in malignant cases
                feature_relevance_score = 0
                total_sequences = len(malignant_df)
                
                if total_sequences > 0:
                    for important_feature in important_features:
                        count = sum(1 for sequence in malignant_df['sequence'] 
                                   if pd.notna(sequence) and important_feature in sequence)
                        feature_relevance_score += count / total_sequences
                    
                    validation_results[method] = {
                        'biological_relevance_score': feature_relevance_score / len(important_features),
                        'method': method
                    }
            except FileNotFoundError:
                print(f"Warning: File 'data/Cancer_Data_sequences_{method}.csv' not found for validation.")
                continue
        
        return validation_results

    def compare_classification_performance(self):
        """Compare classification performance across different binning methods"""
        from collections import Counter
        
        performance_results = {}
        
        for method in ['quantile', 'uniform', 'kmeans']:
            try:
                df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
                
                # Create simple features from sequences
                feature_vectors = []
                labels = []
                
                # Extract unique features across all sequences
                all_features = set()
                for sequence in df['sequence']:
                    if pd.notna(sequence):
                        all_features.update(sequence.split())
                
                all_features = sorted(list(all_features))
                
                # Convert sequences to binary feature vectors
                for idx, row in df.iterrows():
                    sequence = row['sequence']
                    if pd.isna(sequence):
                        continue
                        
                    features = sequence.split()
                    vector = [1 if feature in features else 0 for feature in all_features]
                    feature_vectors.append(vector)
                    labels.append(1 if row['diagnosis'] == 'Malignant' else 0)
                
                if len(feature_vectors) > 10:  # Ensure we have enough samples
                    X = np.array(feature_vectors)
                    y = np.array(labels)
                    
                    # Cross-validation
                    clf = RandomForestClassifier(n_estimators=50, random_state=42)
                    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                    
                    performance_results[method] = {
                        'mean_cv_accuracy': cv_scores.mean(),
                        'std_cv_accuracy': cv_scores.std(),
                        'n_features': len(all_features),
                        'n_samples': len(feature_vectors)
                    }
                
            except FileNotFoundError:
                print(f"Warning: File 'data/Cancer_Data_sequences_{method}.csv' not found for performance analysis.")
                continue
            except Exception as e:
                print(f"Error processing {method}: {str(e)}")
                continue
        
        return performance_results

    def analyze_sequence_diversity(self):
        """Analyze diversity of sequences across methods"""
        diversity_results = {}
        
        for method in ['quantile', 'uniform', 'kmeans']:
            try:
                df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
                
                # Calculate sequence diversity metrics
                unique_sequences = df['sequence'].nunique()
                total_sequences = len(df)
                
                # Calculate feature usage distribution
                all_features = []
                for sequence in df['sequence']:
                    if pd.notna(sequence):
                        all_features.extend(sequence.split())
                
                from collections import Counter
                feature_counter = Counter(all_features)
                
                # Shannon diversity index for feature usage
                total_features = sum(feature_counter.values())
                shannon_diversity = -sum((count/total_features) * np.log(count/total_features) 
                                       for count in feature_counter.values() if count > 0)
                
                diversity_results[method] = {
                    'unique_sequences': unique_sequences,
                    'total_sequences': total_sequences,
                    'sequence_diversity_ratio': unique_sequences / total_sequences,
                    'feature_shannon_diversity': shannon_diversity,
                    'total_unique_features': len(feature_counter)
                }
                
            except FileNotFoundError:
                print(f"Warning: File 'data/Cancer_Data_sequences_{method}.csv' not found for diversity analysis.")
                continue
        
        return diversity_results

# Performance comparison
def compare_methods():
    analyzer = SensitivityAnalyzer()
    
    print("=== Sensitivity Analysis Results ===\n")
    
    # Check if sequence files exist
    methods = ['quantile', 'uniform', 'kmeans']
    existing_files = []
    for method in methods:
        try:
            df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
            existing_files.append(method)
            print(f"✓ Found: data/Cancer_Data_sequences_{method}.csv ({len(df)} sequences)")
        except FileNotFoundError:
            print(f"✗ Missing: data/Cancer_Data_sequences_{method}.csv")
    
    if not existing_files:
        print("ERROR: No sequence files found in data/ folder.")
        return
    
    print(f"\nAnalyzing {len(existing_files)} available methods: {existing_files}\n")
    
    # Analyze patterns
    pattern_results = analyzer.analyze_binning_methods()
    
    for method, results in pattern_results.items():
        print(f"Method: {method.upper()}")
        print(f"  Total sequences: {results['total_sequences']}")
        print(f"  Avg sequence length: {results['avg_sequence_length']:.2f}")
        print(f"  Top malignant patterns:")
        for feature, count in results['malignant_patterns'][:5]:
            print(f"    {feature}: {count}")
        print(f"  Top benign patterns:")
        for feature, count in results['benign_patterns'][:3]:
            print(f"    {feature}: {count}")
        print()
    
    # Biological relevance validation
    print("=== Biological Relevance Validation ===\n")
    validation_results = analyzer.validate_biological_relevance()
    
    for method, results in validation_results.items():
        print(f"{method.capitalize()}: {results['biological_relevance_score']:.3f}")
    
    # Classification performance comparison
    print("\n=== Classification Performance Comparison ===\n")
    performance_results = analyzer.compare_classification_performance()
    
    for method, results in performance_results.items():
        print(f"{method.capitalize()}:")
        print(f"  CV Accuracy: {results['mean_cv_accuracy']:.3f} (±{results['std_cv_accuracy']:.3f})")
        print(f"  Features: {results['n_features']}")
        print(f"  Samples: {results['n_samples']}")
        print()
    
    # Sequence diversity analysis
    print("=== Sequence Diversity Analysis ===\n")
    diversity_results = analyzer.analyze_sequence_diversity()
    
    for method, results in diversity_results.items():
        print(f"{method.capitalize()}:")
        print(f"  Unique sequences: {results['unique_sequences']}/{results['total_sequences']} ({results['sequence_diversity_ratio']:.3f})")
        print(f"  Feature diversity: {results['feature_shannon_diversity']:.3f}")
        print(f"  Unique features: {results['total_unique_features']}")
        print()

if __name__ == "__main__":
    compare_methods()