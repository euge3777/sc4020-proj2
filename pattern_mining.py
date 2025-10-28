from collections import Counter, defaultdict
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class SequentialPatternMiner:
    def __init__(self, min_support=0.1):
        self.min_support = min_support
        self.patterns = {}
    
    def mine_patterns(self, sequences, diagnosis_labels):
        """Mine sequential patterns for each diagnosis class"""
        
        # Separate by diagnosis
        malignant_sequences = [seq for seq, diag in zip(sequences, diagnosis_labels) 
                             if diag == 'Malignant']
        benign_sequences = [seq for seq, diag in zip(sequences, diagnosis_labels) 
                          if diag == 'Benign']
        
        print(f"Mining patterns from {len(malignant_sequences)} malignant and {len(benign_sequences)} benign sequences")
        
        # Mine patterns for each class
        self.patterns['Malignant'] = self._find_frequent_patterns(malignant_sequences)
        self.patterns['Benign'] = self._find_frequent_patterns(benign_sequences)
        
        return self.patterns
    
    def _find_frequent_patterns(self, sequences):
        """Find frequent sequential patterns"""
        # Convert sequences to list of features
        feature_sequences = [seq.split() for seq in sequences]
        
        # Find frequent individual items
        item_counts = Counter()
        for seq in feature_sequences:
            item_counts.update(seq)
        
        min_count = len(sequences) * self.min_support
        frequent_items = {item: count for item, count in item_counts.items() 
                         if count >= min_count}
        
        # Find frequent pairs (2-grams)
        pair_counts = Counter()
        for seq in feature_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += 1
        
        frequent_pairs = {pair: count for pair, count in pair_counts.items() 
                         if count >= min_count}
        
        return {
            'frequent_items': frequent_items,
            'frequent_pairs': frequent_pairs,
            'total_sequences': len(sequences)
        }
    
    def generate_pattern_features(self, sequences):
        """Generate binary features based on discovered patterns"""
        feature_vectors = []
        
        # Collect all unique patterns
        all_items = set()
        all_pairs = set()
        
        for diagnosis in self.patterns:
            all_items.update(self.patterns[diagnosis]['frequent_items'].keys())
            all_pairs.update(self.patterns[diagnosis]['frequent_pairs'].keys())
        
        # Convert sequences to binary feature vectors
        for seq in sequences:
            features = seq.split()
            vector = {}
            
            # Individual item features
            for item in all_items:
                vector[f"has_{item}"] = 1 if item in features else 0
            
            # Pair features
            for i in range(len(features) - 1):
                pair = (features[i], features[i+1])
                for target_pair in all_pairs:
                    if pair == target_pair:
                        vector[f"has_pair_{pair[0]}_then_{pair[1]}"] = 1
            
            # Fill missing pair features with 0
            for pair in all_pairs:
                pair_key = f"has_pair_{pair[0]}_then_{pair[1]}"
                if pair_key not in vector:
                    vector[pair_key] = 0
            
            feature_vectors.append(vector)
        
        return feature_vectors

def run_pattern_mining_analysis():
    """Run complete pattern mining analysis on all three methods"""
    
    methods = ['quantile', 'uniform', 'kmeans']
    results = {}
    
    for method in methods:
        print(f"\n=== Pattern Mining for {method.upper()} method ===")
        
        # Load data
        df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
        
        # Mine patterns
        miner = SequentialPatternMiner(min_support=0.15)
        patterns = miner.mine_patterns(df['sequence'].tolist(), 
                                     df['diagnosis'].tolist())
        
        # Print discovered patterns
        print(f"\nMalignant patterns:")
        for item, count in sorted(patterns['Malignant']['frequent_items'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {item}: {count}")
        
        print(f"\nBenign patterns:")
        for item, count in sorted(patterns['Benign']['frequent_items'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {item}: {count}")
        
        # Generate features and test classifier
        feature_vectors = miner.generate_pattern_features(df['sequence'].tolist())
        
        if feature_vectors:  # Check if we have features
            # Convert to DataFrame for easier handling
            feature_df = pd.DataFrame(feature_vectors)
            X = feature_df.values
            y = df['diagnosis'].map({'Malignant': 1, 'Benign': 0}).values
            
            # Train classifier
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            accuracy = clf.score(X_test, y_test)
            print(f"\nClassification accuracy: {accuracy:.3f}")
            
            results[method] = {
                'patterns': patterns,
                'accuracy': accuracy,
                'n_features': X.shape[1]
            }
    
    return results

if __name__ == "__main__":
    import pandas as pd
    results = run_pattern_mining_analysis()