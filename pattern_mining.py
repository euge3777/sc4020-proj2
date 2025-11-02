from collections import Counter, defaultdict
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class SequentialPatternMiner:
    def __init__(self, min_support=0.1, max_pattern_length=5):
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.patterns = {}
    
    def mine_patterns(self, sequences, diagnosis_labels):
        """Mine sequential patterns for each diagnosis class using GSP algorithm"""
        
        # Separate by diagnosis
        malignant_sequences = [seq for seq, diag in zip(sequences, diagnosis_labels) 
                             if diag == 'Malignant']
        benign_sequences = [seq for seq, diag in zip(sequences, diagnosis_labels) 
                          if diag == 'Benign']
        
        print(f"Mining patterns from {len(malignant_sequences)} malignant and {len(benign_sequences)} benign sequences")
        
        # Mine patterns for each class using GSP
        self.patterns['Malignant'] = self._gsp_mine(malignant_sequences)
        self.patterns['Benign'] = self._gsp_mine(benign_sequences)
        
        return self.patterns
    
    def _gsp_mine(self, sequences):
        """Generalized Sequential Pattern (GSP) algorithm"""
        # Convert sequences to list of features
        feature_sequences = [seq.split() for seq in sequences]
        
        min_count = len(sequences) * self.min_support
        
        # Level 1: Find frequent 1-sequences (single items)
        item_counts = Counter()
        for seq in feature_sequences:
            unique_items = set(seq)  # Count each item once per sequence
            item_counts.update(unique_items)
        
        # Filter frequent 1-sequences
        frequent_patterns = {
            1: {(item,): count for item, count in item_counts.items() if count >= min_count}
        }
        
        if not frequent_patterns[1]:
            return {
                'sequential_patterns': {},
                'total_sequences': len(sequences),
                'max_length': 0
            }
        
        # Generate longer patterns iteratively
        k = 2
        while k <= self.max_pattern_length and frequent_patterns.get(k-1):
            # Generate candidate k-sequences
            candidates = self._generate_candidates(frequent_patterns[k-1])
            
            # Count support for each candidate
            candidate_counts = Counter()
            for seq in feature_sequences:
                for candidate in candidates:
                    if self._is_subsequence(candidate, seq):
                        candidate_counts[candidate] += 1
            
            # Filter by minimum support
            frequent_k = {pattern: count for pattern, count in candidate_counts.items() 
                         if count >= min_count}
            
            if not frequent_k:
                break
            
            frequent_patterns[k] = frequent_k
            k += 1
        
        # Flatten all patterns and sort by support
        all_patterns = {}
        for length, patterns in frequent_patterns.items():
            all_patterns.update(patterns)
        
        return {
            'sequential_patterns': all_patterns,
            'total_sequences': len(sequences),
            'max_length': max(frequent_patterns.keys()) if frequent_patterns else 0
        }
    
    def _generate_candidates(self, frequent_patterns):
        """Generate candidate (k+1)-sequences from frequent k-sequences"""
        candidates = set()
        pattern_list = list(frequent_patterns.keys())
        
        for i, p1 in enumerate(pattern_list):
            for p2 in pattern_list[i:]:
                # Join patterns if they share k-1 prefix/suffix
                # Method 1: Append last item of p2 to p1
                new_pattern = p1 + (p2[-1],)
                candidates.add(new_pattern)
                
                # Method 2: Append last item of p1 to p2 (if different)
                if p1 != p2:
                    new_pattern = p2 + (p1[-1],)
                    candidates.add(new_pattern)
        
        return candidates
    
    def _is_subsequence(self, pattern, sequence):
        """Check if pattern is a subsequence of sequence (maintains order)"""
        pattern_idx = 0
        for item in sequence:
            if pattern_idx < len(pattern) and item == pattern[pattern_idx]:
                pattern_idx += 1
                if pattern_idx == len(pattern):
                    return True
        return pattern_idx == len(pattern)
    
    def generate_pattern_features(self, sequences):
        """Generate binary features based on discovered sequential patterns"""
        feature_vectors = []
        
        # Collect all unique sequential patterns from both classes
        all_patterns = set()
        for diagnosis in self.patterns:
            all_patterns.update(self.patterns[diagnosis]['sequential_patterns'].keys())
        
        # Convert sequences to binary feature vectors
        for seq in sequences:
            features = seq.split()
            vector = {}
            
            # Check each pattern
            for pattern in all_patterns:
                pattern_key = '_then_'.join(pattern)
                vector[f"has_pattern_{pattern_key}"] = 1 if self._is_subsequence(pattern, features) else 0
            
            feature_vectors.append(vector)
        
        return feature_vectors

def run_pattern_mining_analysis():
    """Run complete pattern mining analysis on all three methods using GSP"""
    import pandas as pd
    
    methods = ['quantile', 'uniform', 'kmeans']
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Pattern Mining for {method.upper()} method (GSP Algorithm)")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_csv(f'data/Cancer_Data_sequences_{method}.csv')
        
        # Mine patterns using GSP
        miner = SequentialPatternMiner(min_support=0.15, max_pattern_length=5)
        patterns = miner.mine_patterns(df['sequence'].tolist(), 
                                     df['diagnosis'].tolist())
        
        # Print discovered sequential patterns for Malignant
        print(f"\nTop 15 Malignant Sequential Patterns:")
        malignant_patterns = patterns['Malignant']['sequential_patterns']
        sorted_malignant = sorted(malignant_patterns.items(), 
                                 key=lambda x: (len(x[0]), x[1]), reverse=True)[:15]
        
        for pattern, count in sorted_malignant:
            pattern_str = ' → '.join(pattern)
            support = count / patterns['Malignant']['total_sequences']
            print(f"  [{len(pattern)}] {pattern_str}")
            print(f"      Support: {count}/{patterns['Malignant']['total_sequences']} ({support:.2%})")
        
        # Print discovered sequential patterns for Benign
        print(f"\nTop 15 Benign Sequential Patterns:")
        benign_patterns = patterns['Benign']['sequential_patterns']
        sorted_benign = sorted(benign_patterns.items(), 
                              key=lambda x: (len(x[0]), x[1]), reverse=True)[:15]
        
        for pattern, count in sorted_benign:
            pattern_str = ' → '.join(pattern)
            support = count / patterns['Benign']['total_sequences']
            print(f"  [{len(pattern)}] {pattern_str}")
            print(f"      Support: {count}/{patterns['Benign']['total_sequences']} ({support:.2%})")
        
        # Print summary statistics
        print(f"\nPattern Statistics:")
        print(f"  Malignant: {len(malignant_patterns)} patterns, max length: {patterns['Malignant']['max_length']}")
        print(f"  Benign: {len(benign_patterns)} patterns, max length: {patterns['Benign']['max_length']}")
        
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
            print(f"Number of pattern features: {X.shape[1]}")
            
            results[method] = {
                'patterns': patterns,
                'accuracy': accuracy,
                'n_features': X.shape[1]
            }
    
    return results

if __name__ == "__main__":
    results = run_pattern_mining_analysis()