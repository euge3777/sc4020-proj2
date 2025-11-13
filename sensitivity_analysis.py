import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pattern_mining import SequentialPatternMiner

class SensitivityAnalyzer:
    def __init__(self, min_support=0.15, max_pattern_length=5):
        self.results = {}
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
    
    def analyze_binning_methods(self):
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
        # Mine sequential patterns using GSP
        miner = SequentialPatternMiner(
            min_support=self.min_support,
            max_pattern_length=self.max_pattern_length
        )
        
        patterns = miner.mine_patterns(
            df['sequence'].tolist(),
            df['diagnosis'].tolist()
        )
        
        # Extract top patterns for each class
        malignant_patterns = patterns['Malignant']['sequential_patterns']
        benign_patterns = patterns['Benign']['sequential_patterns']
        
        # Sort by length and support
        malignant_sorted = sorted(
            malignant_patterns.items(),
            key=lambda x: (len(x[0]), x[1]),
            reverse=True
        )[:15]
        
        benign_sorted = sorted(
            benign_patterns.items(),
            key=lambda x: (len(x[0]), x[1]),
            reverse=True
        )[:15]
        
        results = {
            'method': method_name,
            'total_sequences': len(df),
            'avg_sequence_length': df['sequence_length'].mean(),
            'malignant_patterns': malignant_sorted,
            'benign_patterns': benign_sorted,
            'malignant_max_length': patterns['Malignant']['max_length'],
            'benign_max_length': patterns['Benign']['max_length'],
            'malignant_total_patterns': len(malignant_patterns),
            'benign_total_patterns': len(benign_patterns)
        }
        
        return results
    
    def validate_biological_relevance(self):
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
        performance_results = {}
        
        for method in ['quantile', 'uniform', 'kmeans']:
            try:
                df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
                
                # Mine patterns using GSP
                miner = SequentialPatternMiner(
                    min_support=self.min_support,
                    max_pattern_length=self.max_pattern_length
                )
                
                patterns = miner.mine_patterns(
                    df['sequence'].tolist(),
                    df['diagnosis'].tolist()
                )
                
                # Generate pattern-based features
                feature_vectors = miner.generate_pattern_features(df['sequence'].tolist())
                
                if feature_vectors:
                    # Convert to array
                    feature_df = pd.DataFrame(feature_vectors)
                    X = feature_df.values
                    y = df['diagnosis'].map({'Malignant': 1, 'Benign': 0}).values
                    
                    # Cross-validation with pattern features
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                    
                    # Calculate pattern statistics
                    malignant_patterns = patterns['Malignant']['sequential_patterns']
                    benign_patterns = patterns['Benign']['sequential_patterns']
                    
                    # Longest patterns
                    longest_malignant = max([len(p) for p in malignant_patterns.keys()]) if malignant_patterns else 0
                    longest_benign = max([len(p) for p in benign_patterns.keys()]) if benign_patterns else 0
                    
                    performance_results[method] = {
                        'mean_cv_accuracy': cv_scores.mean(),
                        'std_cv_accuracy': cv_scores.std(),
                        'n_pattern_features': X.shape[1],
                        'n_samples': len(feature_vectors),
                        'malignant_patterns_count': len(malignant_patterns),
                        'benign_patterns_count': len(benign_patterns),
                        'longest_malignant_pattern': longest_malignant,
                        'longest_benign_pattern': longest_benign
                    }
                
            except FileNotFoundError:
                print(f"Warning: File 'data/Cancer_Data_sequences_{method}.csv' not found for performance analysis.")
                continue
            except Exception as e:
                print(f"Error processing {method}: {str(e)}")
                continue
        
        return performance_results

    def analyze_sequence_diversity(self):
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
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print(f"{'='*60}")
        print(f"  Total sequences: {results['total_sequences']}")
        print(f"  Avg sequence length: {results['avg_sequence_length']:.2f}")
        print(f"  Total patterns found:")
        print(f"    Malignant: {results['malignant_total_patterns']} (max length: {results['malignant_max_length']})")
        print(f"    Benign: {results['benign_total_patterns']} (max length: {results['benign_max_length']})")
        
        print(f"\n  Top 5 Malignant Sequential Patterns:")
        for pattern, count in results['malignant_patterns'][:5]:
            pattern_str = ' → '.join(pattern)
            support = count / results['total_sequences']
            print(f"    [{len(pattern)}] {pattern_str}")
            print(f"        Support: {count} ({support:.2%})")
        
        print(f"\n  Top 5 Benign Sequential Patterns:")
        for pattern, count in results['benign_patterns'][:5]:
            pattern_str = ' → '.join(pattern)
            support = count / results['total_sequences']
            print(f"    [{len(pattern)}] {pattern_str}")
            print(f"        Support: {count} ({support:.2%})")
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
        print(f"  Pattern Features: {results['n_pattern_features']}")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Patterns Discovered:")
        print(f"    Malignant: {results['malignant_patterns_count']} (longest: {results['longest_malignant_pattern']})")
        print(f"    Benign: {results['benign_patterns_count']} (longest: {results['longest_benign_pattern']})")
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