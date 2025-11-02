"""
Cancer Feature Pattern Mining - Data Preprocessing
SC4020 Assignment 2 - Task 2.2

This script transforms numerical features from the breast cancer dataset into categorical sequences
for sequential pattern mining (e.g., GSP algorithm).

Sequence Semantics:
- Rank features per patient by z-score (standardized feature importance)
- Select top-k features as ordered itemsets
- Maximum sequence length: L
- Max-gap = 1 (allowing same-order ties to form a single itemset)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class CancerSequencePreprocessor:
    """
    Preprocessor for transforming cancer data into sequential patterns.
    """
    
    def __init__(self, data_path, top_k=5, max_seq_length=5, 
                 binning_strategy='quantile', n_bins=3, 
                 feature_types=['mean', 'se', 'worst']):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        data_path : str
            Path to the cancer data CSV file
        top_k : int
            Number of top features to select per patient (default: 5)
        max_seq_length : int
            Maximum sequence length L (default: 5)
        binning_strategy : str
            Strategy for discretization: 'uniform', 'quantile', or 'kmeans' (default: 'quantile')
        n_bins : int
            Number of bins for discretization (default: 3 for low/medium/high)
        feature_types : list of str
            Types of features to include: 'mean', 'se', and/or 'worst' (default: all three)
        """
        self.data_path = data_path
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        self.binning_strategy = binning_strategy
        self.n_bins = n_bins
        self.feature_types = feature_types
        
        # Data containers
        self.df = None
        self.feature_cols = None
        self.X = None
        self.y = None
        
        # Preprocessing objects
        self.scaler = StandardScaler()
        self.discretizers = {}
        
        # Results
        self.X_scaled = None
        self.X_discretized = None
        self.feature_importance = None
        self.sequences = None
        
    def load_data(self):
        """Load and prepare the cancer dataset."""
        print("="*80)
        print("STEP 1: Loading Cancer Data")
        print("="*80)
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded dataset with {len(self.df)} samples")
        
        # Identify feature columns (exclude id and diagnosis)
        all_feature_cols = [col for col in self.df.columns 
                            if col not in ['id', 'diagnosis']]
        
        # Filter features by selected types (mean, se, worst)
        self.feature_cols = []
        for col in all_feature_cols:
            for feature_type in self.feature_types:
                if col.endswith(f'_{feature_type}'):
                    self.feature_cols.append(col)
                    break
        
        print(f"✓ Selected feature types: {self.feature_types}")
        print(f"✓ Identified {len(self.feature_cols)} features (from {len(all_feature_cols)} total)")
        print(f"  Features: {self.feature_cols[:5]}... (showing first 5)")
        
        # Check for missing values
        missing_before = self.df[self.feature_cols].isnull().sum().sum()
        if missing_before > 0:
            print(f"⚠ Found {missing_before} missing values")
            # Drop columns with all NaN or fill missing values
            self.df = self.df.dropna(axis=1, how='all')
            self.feature_cols = [col for col in self.df.columns 
                                if col not in ['id', 'diagnosis']]
            # Fill remaining NaN with median
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(
                self.df[self.feature_cols].median()
            )
            print(f"✓ Cleaned missing values, {len(self.feature_cols)} features remaining")
        
        # Prepare X and y
        self.X = self.df[self.feature_cols].values
        self.y = (self.df['diagnosis'] == 'M').astype(int).values  # 1 for Malignant, 0 for Benign
        
        print(f"✓ Target distribution: {sum(self.y)} Malignant, {len(self.y) - sum(self.y)} Benign")
        print()
        
        return self
    
    def standardize_features(self):
        """Standardize features using z-score normalization."""
        print("="*80)
        print("STEP 2: Z-Score Standardization")
        print("="*80)
        
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"✓ Standardized {self.X_scaled.shape[1]} features")
        print(f"  Mean: {self.X_scaled.mean():.6f} (≈0 expected)")
        print(f"  Std: {self.X_scaled.std():.6f} (≈1 expected)")
        print()
        
        return self
    
    def calculate_feature_importance(self):
        """
        Calculate feature importance using mutual information w.r.t. diagnosis.
        Also provides z-score based ranking per patient.
        """
        print("="*80)
        print("STEP 3: Feature Importance Calculation")
        print("="*80)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        print("✓ Calculated Mutual Information scores")
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10).to_string(index=False))
        print()
        
        return self
    
    def discretize_features(self):
        """
        Discretize continuous features into categorical bins.
        Uses KBinsDiscretizer with the specified strategy.
        """
        print("="*80)
        print(f"STEP 4: Feature Discretization ({self.binning_strategy} strategy)")
        print("="*80)
        
        # Create discretizer
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy=self.binning_strategy
        )
        
        # Fit and transform
        self.X_discretized = discretizer.fit_transform(self.X)
        
        # Store discretizer for each feature
        for i, feature in enumerate(self.feature_cols):
            self.discretizers[feature] = {
                'bin_edges': discretizer.bin_edges_[i],
                'n_bins': self.n_bins
            }
        
        print(f"✓ Discretized features into {self.n_bins} bins")
        print(f"  Strategy: {self.binning_strategy}")
        print(f"  Encoding: ordinal (0={self._get_bin_label(0)}, "
              f"1={self._get_bin_label(1)}, 2={self._get_bin_label(2)})")
        
        # Show example bin edges for first feature
        example_feature = self.feature_cols[0]
        bin_edges = self.discretizers[example_feature]['bin_edges']
        print(f"\n  Example bin edges for '{example_feature}':")
        for i in range(len(bin_edges)-1):
            print(f"    Bin {i} ({self._get_bin_label(i)}): [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})")
        print()
        
        return self
    
    def _get_bin_label(self, bin_idx):
        """Get human-readable label for bin index."""
        if self.n_bins == 3:
            return ['low', 'medium', 'high'][bin_idx]
        elif self.n_bins == 5:
            return ['very_low', 'low', 'medium', 'high', 'very_high'][bin_idx]
        else:
            return f'bin_{bin_idx}'
    
    def create_sequences(self):
        """
        Create sequential patterns for each patient.
        
        For each patient:
        1. Rank features by their z-score (absolute value)
        2. Select top-k features
        3. Order them by z-score rank
        4. Create sequence with discretized values
        5. Group features with same rank (ties) into single itemset
        """
        print("="*80)
        print("STEP 5: Sequential Pattern Creation")
        print("="*80)
        
        sequences = []
        
        for patient_idx in range(len(self.X_scaled)):
            # Get z-scores for this patient
            z_scores = self.X_scaled[patient_idx]
            
            # Rank features by absolute z-score (importance for this patient)
            feature_ranks = np.argsort(np.abs(z_scores))[::-1]  # Descending order
            
            # Select top-k features
            top_features_idx = feature_ranks[:self.top_k]
            
            # Get their z-scores and discretized values
            top_z_scores = z_scores[top_features_idx]
            top_discretized = self.X_discretized[patient_idx, top_features_idx]
            
            # Create sequence items
            sequence_items = []
            for feat_idx, z_score, disc_val in zip(top_features_idx, top_z_scores, top_discretized):
                feature_name = self.feature_cols[feat_idx]
                bin_label = self._get_bin_label(int(disc_val))
                
                sequence_items.append({
                    'feature': feature_name,
                    'z_score': z_score,
                    'abs_z_score': abs(z_score),
                    'bin': int(disc_val),
                    'bin_label': bin_label,
                    'itemset': f"{bin_label}_{feature_name}"
                })
            
            # Sort by absolute z-score (already sorted, but explicit)
            sequence_items.sort(key=lambda x: x['abs_z_score'], reverse=True)
            
            # Limit to max_seq_length
            sequence_items = sequence_items[:self.max_seq_length]
            
            # Store sequence
            sequences.append({
                'patient_id': self.df.iloc[patient_idx]['id'],
                'diagnosis': 'Malignant' if self.y[patient_idx] == 1 else 'Benign',
                'sequence_items': sequence_items,
                'sequence': [item['itemset'] for item in sequence_items]
            })
        
        self.sequences = sequences
        
        print(f"✓ Created sequences for {len(sequences)} patients")
        print(f"  Top-k features per patient: {self.top_k}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print()
        
        return self
    
    def display_sample_sequences(self, n_malignant=5, n_benign=5):
        """Display sample sequences for inspection."""
        print("="*80)
        print("STEP 6: Sample Sequences")
        print("="*80)
        
        # Get malignant samples
        malignant_seqs = [s for s in self.sequences if s['diagnosis'] == 'Malignant']
        benign_seqs = [s for s in self.sequences if s['diagnosis'] == 'Benign']
        
        print(f"\n{'MALIGNANT CASES':-^80}")
        for i, seq in enumerate(malignant_seqs[:n_malignant]):
            print(f"\nPatient {seq['patient_id']}:")
            print(f"  Sequence: <{', '.join(['{' + item + '}' for item in seq['sequence']])}>\n")
            for j, item in enumerate(seq['sequence_items']):
                print(f"    {j+1}. {item['itemset']:<40} (z-score: {item['z_score']:>7.3f})")
        
        print(f"\n\n{'BENIGN CASES':-^80}")
        for i, seq in enumerate(benign_seqs[:n_benign]):
            print(f"\nPatient {seq['patient_id']}:")
            print(f"  Sequence: <{', '.join(['{' + item + '}' for item in seq['sequence']])}>\n")
            for j, item in enumerate(seq['sequence_items']):
                print(f"    {j+1}. {item['itemset']:<40} (z-score: {item['z_score']:>7.3f})")
        
        print("\n")
        
    def save_sequences(self, output_path=None, suffix=None):
        """Save sequences to CSV file for sequential pattern mining."""
        if output_path is None:
            if suffix:
                output_path = self.data_path.replace('.csv', f'_sequences_{suffix}.csv')
            else:
                output_path = self.data_path.replace('.csv', '_sequences.csv')
        
        print("="*80)
        print("STEP 7: Saving Sequences")
        print("="*80)
        
        # Create DataFrame with sequences
        sequence_df = pd.DataFrame([
            {
                'patient_id': seq['patient_id'],
                'diagnosis': seq['diagnosis'],
                'sequence': ' '.join(seq['sequence']),
                'sequence_length': len(seq['sequence'])
            }
            for seq in self.sequences
        ])
        
        sequence_df.to_csv(output_path, index=False)
        print(f"✓ Saved sequences to: {output_path}")
        print(f"  Total sequences: {len(sequence_df)}")
        print(f"  Malignant: {sum(sequence_df['diagnosis'] == 'Malignant')}")
        print(f"  Benign: {sum(sequence_df['diagnosis'] == 'Benign')}")
        print()
        
        return output_path
    
    def get_statistics(self):
        """Generate preprocessing statistics for reporting."""
        print("="*80)
        print("PREPROCESSING STATISTICS")
        print("="*80)
        
        stats = {
            'total_patients': len(self.sequences),
            'total_features': len(self.feature_cols),
            'feature_types': self.feature_types,
            'malignant_count': sum(s['diagnosis'] == 'Malignant' for s in self.sequences),
            'benign_count': sum(s['diagnosis'] == 'Benign' for s in self.sequences),
            'binning_strategy': self.binning_strategy,
            'n_bins': self.n_bins,
            'top_k': self.top_k,
            'max_seq_length': self.max_seq_length,
            'avg_sequence_length': np.mean([len(s['sequence']) for s in self.sequences])
        }
        
        print(f"Total Patients: {stats['total_patients']}")
        print(f"  - Malignant: {stats['malignant_count']}")
        print(f"  - Benign: {stats['benign_count']}")
        print(f"\nFeature Selection:")
        print(f"  - Feature types: {', '.join(stats['feature_types'])}")
        print(f"  - Total features: {stats['total_features']}")
        print(f"\nDiscretization:")
        print(f"  - Strategy: {stats['binning_strategy']}")
        print(f"  - Number of bins: {stats['n_bins']}")
        print(f"\nSequence Parameters:")
        print(f"  - Top-k features: {stats['top_k']}")
        print(f"  - Max sequence length: {stats['max_seq_length']}")
        print(f"  - Average sequence length: {stats['avg_sequence_length']:.2f}")
        print()
        
        return stats
    
    def run_full_preprocessing(self, suffix=None):
        """Execute the complete preprocessing pipeline."""
        self.load_data()
        self.standardize_features()
        self.calculate_feature_importance()
        self.discretize_features()
        self.create_sequences()
        self.display_sample_sequences()
        output_file = self.save_sequences(suffix=suffix)
        self.get_statistics()
        
        return output_file


def sensitivity_analysis_binning_strategies(feature_types=['mean', 'se', 'worst']):
    """
    Perform sensitivity analysis across different binning strategies.
    This addresses the tip: "Use KBinsDiscretizer and report a sensitivity check 
    across binning strategies."
    
    Parameters:
    -----------
    feature_types : list of str
        Types of features to include: 'mean', 'se', and/or 'worst' (default: all three)
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Binning Strategies")
    print(f"Feature types: {', '.join(feature_types)}")
    print("="*80 + "\n")
    
    data_path = r'/Users/liekzhewong/SC4020/SC4020-A2/Cancer_Data.csv'
    strategies = ['uniform', 'quantile', 'kmeans']
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Testing Binning Strategy: {strategy.upper()}")
        print(f"{'='*80}\n")
        
        preprocessor = CancerSequencePreprocessor(
            data_path=data_path,
            top_k=5,
            max_seq_length=5,
            binning_strategy=strategy,
            n_bins=3,
            feature_types=feature_types
        )
        
        output_file = preprocessor.run_full_preprocessing(suffix=strategy)
        
        results[strategy] = {
            'preprocessor': preprocessor,
            'output_file': output_file,
            'stats': preprocessor.get_statistics()
        }
        
        print(f"\n{'='*80}\n")
    
    # Summary comparison
    print("\n" + "="*80)
    print("BINNING STRATEGY COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    comparison_df = pd.DataFrame([
        {
            'Strategy': strategy,
            'Output File': results[strategy]['output_file'].split('\\')[-1],
            'Features': len(results[strategy]['stats']['feature_types']),
            'Total Features': results[strategy]['stats']['total_features'],
            'Avg Seq Length': f"{results[strategy]['stats']['avg_sequence_length']:.2f}",
            'Malignant': results[strategy]['stats']['malignant_count'],
            'Benign': results[strategy]['stats']['benign_count']
        }
        for strategy in strategies
    ])
    
    print(comparison_df.to_string(index=False))
    print("\n")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CANCER FEATURE PATTERN MINING - DATA PREPROCESSING")
    print("SC4020 Assignment 2 - Task 2.2")
    print("="*80 + "\n")
    
    # Example 1: Use only 'mean' features
    # results = sensitivity_analysis_binning_strategies(feature_types=['mean'])
    
    # Example 2: Use 'mean' and 'worst' features
    # results = sensitivity_analysis_binning_strategies(feature_types=['mean', 'worst'])
    
    # Example 3: Use only 'se' features
    # results = sensitivity_analysis_binning_strategies(feature_types=['se'])
    
    # Default: Run sensitivity analysis with all feature types (mean, se, worst)
    results = sensitivity_analysis_binning_strategies(feature_types=['mean'])
    
    print("="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    for strategy, result in results.items():
        print(f"  - {result['output_file']}")
    
    print("\nNext Steps:")
    print("  1. Review the generated sequence files")
    print("  2. Apply GSP (Generalized Sequential Pattern) algorithm")
    print("  3. Compare patterns across binning strategies")
    print("  4. Analyze malignant vs benign pattern differences")
    print("\nCustomization Options:")
    print("  - Change feature_types parameter to select different feature types")
    print("  - Options: ['mean'], ['se'], ['worst'], or any combination")
    print()
