import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import graphviz
from datetime import datetime
import pdb
import seaborn as sns
import random
palette = sns.color_palette("Set2", 19).as_hex()
class BehaviorExtractor:
    def __init__(self):
        self.behavior_classifications = dict()
        self.cleaned_dfs = {}
        self.behavior_mappings = pd.read_csv('behaviors.csv')

    def extract_behaviors(self, file_path, columns=['frame', 'time', 'action']):
        behaviors = []
        string = ''
        in_full_log = False
        start_line = None
        
        with open(file_path, 'r') as file:
            data = file.readlines()

            for i, line in enumerate(data):
                if 'FULL\tLOG' in line:
                    in_full_log = True
                    start_line = i
                    break
        for line in data[start_line+4:]:
            if '-' in line:
                break
            line = line.replace('either', '')
            parts = line.split()
            behaviors.append([parts[0], parts[1], ' '.join(parts[2:])])
        df = pd.DataFrame(behaviors, columns=columns)
        return self.apply_behavior_mappings(df)

    def extract_folder(self, folder_path, columns=['frame', 'time', 'action']):
        for folder in os.listdir(folder_path):
            self.cleaned_dfs[folder] = {}
            for file in os.listdir(os.path.join(folder_path, folder)):
                if file.endswith('.tsv'):
                    file_path = os.path.join(folder_path, folder, file)
                    self.cleaned_dfs[folder][file] = self.extract_behaviors(file_path, columns)
                    self.cleaned_dfs[folder][file]['folder'] = folder
                    self.cleaned_dfs[folder][file]['file'] = file.split('.')[0]

    def get_index(self, group, ix):
        # numerate all folders and files by 'group' and 'ix'
        files = list(self.cleaned_dfs[group].keys())
        print(f"Group: {group}")
        print(f"Index: {ix}")
        print(f"File: {files[ix]}")
        return self.cleaned_dfs[group][files[ix]]

    def behaviors(self, group, ix=0, file=None):
        if file:
            return self.cleaned_dfs[group][file]
        return self.get_index(group, ix)

    def get_time_in_state(self, df):
        """Calculate time spent in each behavior state for a single dataframe"""
        # Convert time strings to seconds for calculation
        df = df.copy()
        df['seconds'] = df['time'].apply(lambda x:
            int(x.split(':')[0])*60 + float(x.split(':')[1]))

        # Calculate time differences
        df['duration'] = df['seconds'].shift(-1) - df['seconds']

        # Group by behavior and sum durations
        time_in_state = df.groupby('action')['duration'].sum()
        return time_in_state

    def analyze_time_in_states(self, group=None, file=None):
        """Analyze time spent in behaviors across specified data"""
        all_times = []

        if group and file:
            # Single file analysis
            df = self.behaviors(group, file=file)
            times = self.get_time_in_state(df)
            times = pd.DataFrame(times).assign(group=group, file=file)
            all_times.append(times)
        elif group:
            # Group analysis
            for file in self.cleaned_dfs[group].keys():
                df = self.behaviors(group, file=file)
                times = self.get_time_in_state(df)
                times = pd.DataFrame(times).assign(group=group, file=file)
                all_times.append(times)
                # print(f'group: {group}, file: {file}')
        else:
            # All data analysis
            for group in self.cleaned_dfs.keys():
                for file in self.cleaned_dfs[group].keys():
                    df = self.behaviors(group, file=file)
                    times = self.get_time_in_state(df)
                    times = pd.DataFrame(times).assign(group=group, file=file)
                    all_times.append(times)

        return pd.concat(all_times)

    def get_behavior_probabilities(self, df):
        """Calculate probability of each behavior in a single dataframe"""
        behavior_counts = df['action'].value_counts()
        total_behaviors = len(df)
        return behavior_counts / total_behaviors

    def analyze_behavior_probabilities(self, group=None, file=None):
        """Analyze behavior probabilities across specified data"""
        all_probs = []

        if group and file:
            df = self.behaviors(group, file=file)
            probs = self.get_behavior_probabilities(df)
            probs = pd.DataFrame(probs).assign(group=group, file=file)
            all_probs.append(probs)
        elif group:
            for file in self.cleaned_dfs[group].keys():
                df = self.behaviors(group, file=file)
                probs = self.get_behavior_probabilities(df)
                probs = pd.DataFrame(probs).assign(group=group, file=file)
                all_probs.append(probs)
        else:
            for group in self.cleaned_dfs.keys():
                for file in self.cleaned_dfs[group].keys():
                    df = self.behaviors(group, file=file)
                    probs = self.get_behavior_probabilities(df)
                    probs = pd.DataFrame(probs).assign(group=group, file=file)
                    all_probs.append(probs)

        return pd.concat(all_probs)

    def get_transition_matrix(self, df):
        """Create transition matrix for behaviors in a single dataframe"""
        # Get unique behaviors
        behaviors = df['action'].unique()

        # Initialize transition matrix
        transitions = pd.DataFrame(0,
                                 index=behaviors,
                                 columns=behaviors)

        # Count transitions
        for i in range(len(df)-1):
            current_behavior = df['action'].iloc[i]
            next_behavior = df['action'].iloc[i+1]
            transitions.loc[current_behavior, next_behavior] += 1

        return transitions

    def get_probability_transition_matrix(self, transition_matrix):
        """Convert count transition matrix to probability transition matrix"""
        prob_matrix = transition_matrix.copy()
        row_sums = prob_matrix.sum(axis=1)
        prob_matrix = prob_matrix.div(row_sums, axis=0)
        return prob_matrix

    def analyze_transitions(self, group=None, file=None):
        """Analyze transitions across specified data"""
        all_transitions = []
        all_prob_transitions = []

        if group and file:
            df = self.behaviors(group, file=file)
            trans = self.get_transition_matrix(df)
            prob_trans = self.get_probability_transition_matrix(trans)
            all_transitions.append((group, file, trans))
            all_prob_transitions.append((group, file, prob_trans))
        elif group:
            for file in self.cleaned_dfs[group].keys():
                df = self.behaviors(group, file=file)
                trans = self.get_transition_matrix(df)
                prob_trans = self.get_probability_transition_matrix(trans)
                all_transitions.append((group, file, trans))
                all_prob_transitions.append((group, file, prob_trans))
        else:
            for group in self.cleaned_dfs.keys():
                for file in self.cleaned_dfs[group].keys():
                    df = self.behaviors(group, file=file)
                    trans = self.get_transition_matrix(df)
                    prob_trans = self.get_probability_transition_matrix(trans)
                    all_transitions.append((group, file, trans))
                    all_prob_transitions.append((group, file, prob_trans))

        return {'trans': all_transitions, 'trans_prob':all_prob_transitions}

    def export_analyses(self, output_dir, group=None, file=None):
        """Export all analyses to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # Export time in states
        time_df = self.analyze_time_in_states(group, file)
        time_df.to_csv(os.path.join(output_dir, 'time_in_states.csv'))

        # Export behavior probabilities
        prob_df = self.analyze_behavior_probabilities(group, file)
        prob_df.to_csv(os.path.join(output_dir, 'behavior_probabilities.csv'))

        # Export transition matrices
        transitions, prob_transitions = self.analyze_transitions(group, file)

        # Create directory for transition matrices
        trans_dir = os.path.join(output_dir, 'transition_matrices')
        prob_trans_dir = os.path.join(output_dir, 'probability_transition_matrices')
        os.makedirs(trans_dir, exist_ok=True)
        os.makedirs(prob_trans_dir, exist_ok=True)

        # Export each transition matrix
        for group_name, file_name, matrix in transitions:
            filename = f"{group_name}_{file_name}_transitions.csv"
            matrix.to_csv(os.path.join(trans_dir, filename))

        # Export each probability transition matrix
        for group_name, file_name, matrix in prob_transitions:
            filename = f"{group_name}_{file_name}_prob_transitions.csv"
            matrix.to_csv(os.path.join(prob_trans_dir, filename))

    def get_file_at_index(self, ix, group=None):
        """
        Get group and file based on index.
        If group is specified, returns file at index for that group.
        If no group, returns group and file from global index.

        Parameters:
        -----------
        ix : int
            Index to access
        group : str, optional
            Group name to index within. If None, indexes across all groups

        Returns:
        --------
        tuple : (group_name, file_name)
            The group and file names at the specified index
        """
        if group:
            # If group specified, get file at index from that group
            if group not in self.cleaned_dfs:
                raise ValueError(f"Group '{group}' not found")

            files = sorted(list(self.cleaned_dfs[group].keys()))
            if ix >= len(files):
                raise IndexError(f"Index {ix} out of range for group '{group}' with {len(files)} files")

            return group, files[ix]
        else:
            # If no group specified, create flat list of all group/file combinations
            all_combinations = []
            for g in sorted(self.cleaned_dfs.keys()):
                for f in sorted(self.cleaned_dfs[g].keys()):
                    all_combinations.append((g, f))

            if ix >= len(all_combinations):
                raise IndexError(f"Index {ix} out of range for {len(all_combinations)} total files")

            return all_combinations[ix]

    def list_all_files(self):
        """
        Print all available files with their indices
        """
        print("\nGlobal indices:")
        print("--------------")
        idx = 0
        for group in sorted(self.cleaned_dfs.keys()):
            for file in sorted(self.cleaned_dfs[group].keys()):
                print(f"{idx}: {group}/{file}")
                idx += 1

        print("\nGroup-specific indices:")
        print("---------------------")
        for group in sorted(self.cleaned_dfs.keys()):
            print(f"\n{group}:")
            for idx, file in enumerate(sorted(self.cleaned_dfs[group].keys())):
                print(f"  {idx}: {file}")

    def get_behavior_summary_stats(self, group=None):
        """
        Calculate summary statistics for behaviors across files within a group
        or across all groups.

        Parameters:
        -----------
        group : str, optional
            Group name to analyze. If None, analyzes across all groups

        Returns:
        --------
        dict containing:
            'mean' : average time in each behavior
            'std' : standard deviation of time in each behavior
            'sem' : standard error of mean
            'total' : total time in each behavior
            'percent' : percentage of total time in each behavior
            'n_files' : number of files included in analysis
        """
        # Get raw time data
        time_data = self.analyze_time_in_states(group=group)

        # Pivot the data to get behaviors as columns for easier analysis
        pivoted_data = time_data.reset_index().pivot(
            columns='action',
            values='duration',
            index=['group', 'file']
        ).fillna(0)

        # Calculate summary statistics
        stats = {
            'mean': pivoted_data.mean(),
            'std': pivoted_data.std(),
            'sem': pivoted_data.sem(),
            'total': pivoted_data.sum(),
            'n_files': len(pivoted_data)
        }

        # Calculate percentages of total time
        total_time = stats['total'].sum()
        stats['percent'] = (stats['total'] / total_time) * 100

        return stats

    def compare_groups_stats(self):
        """
        Compare behavior statistics across all groups

        Returns:
        --------
        dict containing summary statistics for each group and overall
        """
        # Get stats for each group and overall
        all_stats = {'all': self.get_behavior_summary_stats()}

        for group in self.cleaned_dfs.keys():
            all_stats[group] = self.get_behavior_summary_stats(group)

        return all_stats

    def export_summary_stats(self, output_dir):
        """
        Export summary statistics to CSV files

        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get all stats
        stats = self.compare_groups_stats()

        # Export each type of statistic as a separate CSV
        for stat_type in ['mean', 'std', 'sem', 'total', 'percent']:
            # Create DataFrame with all groups
            stat_df = pd.DataFrame({
                group: group_stats[stat_type]
                for group, group_stats in stats.items()
            })

            # Save to CSV
            filename = f'behavior_{stat_type}.csv'
            stat_df.to_csv(os.path.join(output_dir, filename))

    def print_behavior_summary(self, group=None):
        """
        Print a formatted summary of behavior statistics

        Parameters:
        -----------
        group : str, optional
            Group to summarize. If None, summarizes all data
        """
        stats = self.get_behavior_summary_stats(group)

        group_desc = group if group else "all groups"
        print(f"\nBehavior Summary for {group_desc}")
        print("=" * 50)

        
        behaviors = stats['mean'].index

        for behavior in behaviors:
            print(f"\n{behavior}:")
            print(f"  Mean time: {stats['mean'][behavior]:.2f} ± {stats['sem'][behavior]:.2f} seconds")
            print(f"  Total time: {stats['total'][behavior]:.2f} seconds")
            print(f"  Percentage of total: {stats['percent'][behavior]:.1f}%")

    def get_transition_summary_stats(self, group=None):
        """
        Calculate summary statistics for transition probabilities across files
        within a group or across all groups.

        Parameters:
        -----------
        group : str, optional
            Group name to analyze. If None, analyzes across all groups

        Returns:
        --------
        dict containing:
            'mean_prob': average probability for each transition
            'std_prob': standard deviation of probabilities
            'sem_prob': standard error of mean
            'total_counts': total number of each transition
            'n_files': number of files included in analysis
        """
        # Get transitions data
        transitions_data = self.analyze_transitions(group)
        prob_transitions = transitions_data['trans_prob']

        # Get all unique behaviors to create consistent matrix dimensions
        all_behaviors = set()
        for _, _, matrix in prob_transitions:
            all_behaviors.update(matrix.index)
            all_behaviors.update(matrix.columns)
        all_behaviors = sorted(all_behaviors)

        # Create standardized matrices for all files
        standardized_matrices = []
        for _, _, matrix in prob_transitions:
            # Create full-size matrix with zeros (explicitly as float64)
            full_matrix = pd.DataFrame(0.0,
                                     index=all_behaviors,
                                     columns=all_behaviors,
                                     dtype=np.float64)
            # Fill in existing values
            for idx in matrix.index:
                for col in matrix.columns:
                    full_matrix.loc[idx, col] = float(matrix.loc[idx, col])
            standardized_matrices.append(full_matrix)

        # Stack matrices for analysis
        stacked_matrices = np.stack([m.values for m in standardized_matrices])

        # Calculate statistics
        stats = {
            'mean_prob': pd.DataFrame(np.mean(stacked_matrices, axis=0),
                                    index=all_behaviors,
                                    columns=all_behaviors,
                                    dtype=np.float64),
            'std_prob': pd.DataFrame(np.std(stacked_matrices, axis=0),
                                   index=all_behaviors,
                                   columns=all_behaviors,
                                   dtype=np.float64),
            'sem_prob': pd.DataFrame(np.std(stacked_matrices, axis=0) / np.sqrt(len(standardized_matrices)),
                                   index=all_behaviors,
                                   columns=all_behaviors,
                                   dtype=np.float64),
            'n_files': len(standardized_matrices)
        }

        # Calculate total counts from raw transitions
        raw_transitions = transitions_data['trans']
        total_counts = pd.DataFrame(0,
                                  index=all_behaviors,
                                  columns=all_behaviors,
                                  dtype=np.int64)  # Explicitly int64 for counts

        for _, _, matrix in raw_transitions:
            # Create full-size matrix for this file
            full_matrix = pd.DataFrame(0,
                                     index=all_behaviors,
                                     columns=all_behaviors,
                                     dtype=np.int64)  # Explicitly int64 for counts
            # Fill in existing values
            for idx in matrix.index:
                for col in matrix.columns:
                    full_matrix.loc[idx, col] = int(matrix.loc[idx, col])
            total_counts += full_matrix

        stats['total_counts'] = total_counts

        return stats

    def export_transition_summary_stats(self, output_dir, thresh=0.1):
        """
        Export transition summary statistics to CSV files

        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get all stats
        stats = self.compare_groups_transitions()

        # Export statistics for each group
        for group, group_stats in stats.items():
            group_dir = os.path.join(output_dir, group)
            os.makedirs(group_dir, exist_ok=True)

            # Export each type of statistic
            for stat_type in ['mean_prob', 'std_prob', 'sem_prob', 'total_counts']:
                filename = f'transitions_{stat_type}.csv'
                group_stats[stat_type].to_csv(os.path.join(group_dir, filename))

            # Export a summary file with major transitions
            summary_file = os.path.join(group_dir, 'transition_summary.txt')
            with open(summary_file, 'w') as f:
                mean_probs = group_stats['mean_prob']
                sem_probs = group_stats['sem_prob']
                total_counts = group_stats['total_counts']

                f.write(f"Transition Summary for {group}\n")
                f.write("="*50 + "\n")
                f.write(f"Based on {group_stats['n_files']} files\n\n")

                for from_behavior in mean_probs.index:
                    has_transitions = False
                    for to_behavior in mean_probs.columns:
                        prob = mean_probs.loc[from_behavior, to_behavior]
                        if prob > thresh:  # 10% threshold
                            if not has_transitions:
                                f.write(f"\nFrom {from_behavior}:\n")
                                has_transitions = True
                            sem = sem_probs.loc[from_behavior, to_behavior]
                            count = total_counts.loc[from_behavior, to_behavior]
                            f.write(f"  To {to_behavior}:\n")
                            f.write(f"    Probability: {prob:.1%} ± {sem:.1%}\n")
                            f.write(f"    Total occurrences: {count}\n")

    def print_transition_summary(self, group=None, threshold=0.1):
        """
        Print a formatted summary of transition statistics

        Parameters:
        -----------
        group : str, optional
            Group to summarize. If None, summarizes all data
        threshold : float, optional
            Only print transitions with mean probability above this threshold
        """
        stats = self.get_transition_summary_stats(group)

        group_desc = group if group else "all groups"
        print(f"\nTransition Summary for {group_desc}")
        print("=" * 50)
        print(f"Based on {stats['n_files']} files")
        print(f"\nPrinting transitions where (probability > {threshold:.1%}):")

        mean_probs = stats['mean_prob']
        sem_probs = stats['sem_prob']
        total_counts = stats['total_counts']

        for from_behavior in mean_probs.index:
            significant_transitions = []
            for to_behavior in mean_probs.columns:
                prob = mean_probs.loc[from_behavior, to_behavior]
                if prob > threshold:
                    sem = sem_probs.loc[from_behavior, to_behavior]
                    count = total_counts.loc[from_behavior, to_behavior]
                    significant_transitions.append((to_behavior, prob, sem, count))

            if significant_transitions:
                print(f"\nFrom {from_behavior}:")
                for to_behavior, prob, sem, count in sorted(significant_transitions,
                                                          key=lambda x: x[1],
                                                          reverse=True):
                    print(f"  To {to_behavior}:")
                    print(f"    Probability: {prob:.1%} ± {sem:.1%}")
                    print(f"    Total occurrences: {count}")

    def compare_groups_transitions(self):
        # Get stats for each group and overall
        all_stats = {'all': self.get_transition_summary_stats()}

        for group in self.cleaned_dfs.keys():
            all_stats[group] = self.get_transition_summary_stats(group)

        return all_stats

    def apply_behavior_mappings(self, df):
        """Apply behavior classifications and replacements to the dataframe"""
        df = df.copy()

        # Create mapping dictionaries with lowercase keys
        replace_dict = dict(zip(self.behavior_mappings['Behavior'].str.lower(),
                              self.behavior_mappings['ReplaceWith']))
        class_dict = dict(zip(self.behavior_mappings['Behavior'].str.lower(),
                             self.behavior_mappings['Classification']))

        # Replace behavior names and add classification
        df['original_action'] = df['action']
        # Convert to lowercase for mapping, then map
        df['action'] = df['action'].str.lower().map(replace_dict)
        df['classification'] = df['original_action'].str.lower().map(class_dict)

        return df

    def extract_behaviors(self, file_path, columns=['frame', 'time', 'action']):
        behaviors = []
        string = ''
        in_full_log = False
        start_line = None
        with open(file_path, 'r') as file:
            data = file.readlines()

            for i, line in enumerate(data):
                if 'FULL\tLOG' in line:
                    in_full_log = True
                    start_line = i
                    break
        for line in data[start_line+4:]:
            if '-' in line:
                break
            line = line.replace('either', '')
            parts = line.split()
            behaviors.append([parts[0], parts[1], ' '.join(parts[2:])])
        df = pd.DataFrame(behaviors, columns=columns)
        return self.apply_behavior_mappings(df)

    def get_classification_stats(self, group=None):
        """Calculate summary statistics for behavior classifications"""
        # Get time in states data
        time_data = self.analyze_time_in_states(group=group)
        time_data = time_data.reset_index()
    
        # Create mapping dictionary using ReplaceWith values instead of original Behaviors
        behavior_map = dict(zip(self.behavior_mappings['ReplaceWith'].str.lower(),
                                self.behavior_mappings['Classification']))
    
        # Add classification using lowercase mapping
        time_data['classification'] = time_data['action'].str.lower().map(behavior_map)
    
        # For debugging
        print("Unique actions:", time_data['action'].unique())
        print("Available mappings:", behavior_map.keys())
    
        # Group by classification
        class_stats = time_data.groupby('classification')['duration'].agg([
            'mean', 'std', 'sum'
        ]).round(2)
    
        # Calculate percentages
        total_time = class_stats['sum'].sum()
        class_stats['percent'] = (class_stats['sum'] / total_time * 100).round(2)
    
        return class_stats
    def print_classification_summary(self, group=None):
        """Print a formatted summary of classification statistics"""
        stats = self.get_classification_stats(group)

        group_desc = group if group else "all groups"
        print(f"\nBehavior Classification Summary for {group_desc}")
        print("=" * 50)

        for classification in stats.index:
            print(f"\n{classification.capitalize()}:")
            print(f"  Mean time: {stats.loc[classification, 'mean']:.2f} seconds")
            print(f"  Total time: {stats.loc[classification, 'sum']:.2f} seconds")
            print(f"  Percentage of total: {stats.loc[classification, 'percent']:.1f}%")

    def verify_mappings(self, group=None, file=None):
        if group and file:
            df = self.behaviors(group, file=file)
        else:
            # Get first available file if none specified
            group, file = self.get_file_at_index(0)
            df = self.behaviors(group, file=file)

        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Original': df['original_action'],
            'Mapped To': df['action'],
            'Classification': df['classification']
        }).drop_duplicates()

        print(f"\nBehavior Mapping Verification for {group}/{file}")
        print("=" * 50)
        print("\nUnique behaviors and their mappings:")
        print(comparison.to_string())

        print("\n\nClassification Summary:")
        print("-" * 30)
        class_counts = df.groupby('classification')['action'].nunique()
        for classification, count in class_counts.items():
            print(f"\n{classification.capitalize()}:")
            behaviors = df[df['classification'] == classification]['action'].unique()
            print(f"  Number of behaviors: {count}")
            print(f"  Behaviors: {', '.join(behaviors)}")

    def peek_data(self, group=None, file=None, n=5):
        if group and file:
            df = self.behaviors(group, file=file)
        else:
            group, file = self.get_file_at_index(0)
            df = self.behaviors(group, file=file)

        print(f"\nFirst {n} rows of data from {group}/{file}")
        print("=" * 50)
        print(df.head(n).to_string())

        # Show column names
        print("\nAvailable columns:")
        print(df.columns.tolist())

    def create_markov_visualization(self, group=None):
        # Get transition probabilities and state times
        trans_stats = self.get_transition_summary_stats(group)
        time_stats = self.get_behavior_summary_stats(group)

        # Create directed graph
        G = nx.DiGraph()

        # Color scheme for behavior types
        color_map = {
            'aggressive': '#FFE4E1',  # Misty rose
            'aversive': '#E6E6FA',   # Lavender
            'reproductive': '#F0F8FF'  # Alice blue
        }

        # Create behavior to classification mapping
        behavior_class_map = dict(zip(
            self.behavior_mappings['ReplaceWith'],
            self.behavior_mappings['Classification']
        ))

        # Add nodes with attributes
        for behavior in time_stats['mean'].index:
            if behavior in behavior_class_map:
                behavior_type = behavior_class_map[behavior]
                G.add_node(behavior,
                          size=time_stats['percent'][behavior],
                          color=color_map[behavior_type],
                          text_color=self._get_text_color(behavior))

        # Add edges with probability weights
        prob_matrix = trans_stats['mean_prob']
        for from_behavior in prob_matrix.index:
            for to_behavior in prob_matrix.columns:
                prob = prob_matrix.loc[from_behavior, to_behavior]
                if prob > 0.1 and from_behavior in G.nodes and to_behavior in G.nodes:
                    G.add_edge(from_behavior, to_behavior, weight=prob)

        # Layout optimization
        pos = nx.spring_layout(G, k=2, iterations=50)

        return G, pos

    def _get_text_color(self, behavior, classification=None):
        if "blue" in behavior.lower():
            return "#0A06CC"
        elif "yellow" in behavior.lower():
            return "#FA990A"
        elif classification == "aggressive":
            return "red"
        elif classification == "aversive":
            return "blue"
        return "black"

    def create_markov_subgraphs(self, group=None, threshold=0.1):
        stats = self.get_transition_summary_stats(group)
        mean_probs = stats['mean_prob']
        total_counts = stats['total_counts']

        # Get behavior probabilities for node sizes
        behavior_probs = self.analyze_behavior_probabilities(group)
        if isinstance(behavior_probs, pd.DataFrame):
            # Convert to numeric and then calculate mean
            behavior_probs = behavior_probs.reset_index()
            behavior_probs = behavior_probs.groupby('action')['action'].count() / len(behavior_probs)

        # Create classification-based subgraphs
        subgraphs = {}
        for classification in self.behavior_mappings['Classification'].unique():
            subgraphs[classification] = nx.DiGraph(classification=classification)

        # Color scheme for classifications
        classification_colors = {
            'aggressive': '#FFE6E6',  # Light red
            'submissive': '#E6E6FF',  # Light blue
            'display': '#E6FFE6',     # Light green
            'exploratory': '#FFF2E6'  # Light orange
        }

        # Create behavior to classification mapping dictionary
        behavior_to_class = dict(zip(
            self.behavior_mappings['Behavior'].str.lower(),
            self.behavior_mappings['Classification']
        ))

        # Add nodes and edges to appropriate subgraphs
        for from_behavior in mean_probs.index:
            # Get classification for behavior (with fallback)
            from_class = behavior_to_class.get(from_behavior.lower(), 'unknown')
            if from_class == 'unknown':
                print(f"Warning: No classification found for behavior: {from_behavior}")
                continue

            # Set node attributes
            node_attr = {
                'classification': from_class,
                'probability': float(behavior_probs.loc[from_behavior] if from_behavior in behavior_probs.index else 0),
                'fillcolor': classification_colors.get(from_class, '#FFFFFF'),
                'style': 'filled'
            }

            # Set text color based on behavior and classification
            if 'blue' in from_behavior.lower():
                node_attr['fontcolor'] = '#0A06CC'
            elif 'yellow' in from_behavior.lower():
                node_attr['fontcolor'] = '#FA990A'
            elif from_class == 'aggressive':
                node_attr['fontcolor'] = '#CC0000'
            elif from_class == 'aversive':
                node_attr['fontcolor'] = '#0000CC'
            else:
                node_attr['fontcolor'] = 'black'

            subgraphs[from_class].add_node(from_behavior, **node_attr)

            for to_behavior in mean_probs.columns:
                prob = mean_probs.loc[from_behavior, to_behavior]
                if prob > threshold:
                    to_class = behavior_to_class.get(to_behavior.lower(), 'unknown')
                    if to_class == 'unknown':
                        continue

                    # Add edge to both subgraphs if they're different
                    subgraphs[from_class].add_edge(
                        from_behavior, to_behavior,
                        weight=prob,
                        count=total_counts.loc[from_behavior, to_behavior]
                    )

        return subgraphs

    def create_markov_graph(self, group=None, threshold=0.1, debug=False):
        """
        Diagnostic version to track where behaviors are being filtered out
        """
        print("\n=== Starting Diagnostic Analysis ===")

        # Get transition statistics
        stats = self.get_transition_summary_stats(group)
        mean_probs = stats['mean_prob']
        total_counts = stats['total_counts']

        if debug:
            print("\n1. Transition Matrix Shape:", mean_probs.shape)
            print("Behaviors in transition matrix:", mean_probs.index.tolist())
            print("\nSample of transition probabilities:")
            print(mean_probs.head())

        # Get behavior probabilities
        behavior_probs = self.analyze_behavior_probabilities(group)
        if debug:
            print("\n2. Original behavior_probs shape:", behavior_probs.shape)
            print(behavior_probs.head())

        if isinstance(behavior_probs, pd.DataFrame):
            behavior_probs = behavior_probs.reset_index()
            if debug:
                print("\n3. After reset_index:")
                print(behavior_probs.head())

            behavior_probs = behavior_probs.groupby('action')['action'].count() / len(behavior_probs)
            if debug:
                print("\n4. After groupby:")
                print(behavior_probs.head())

        # Create directed graph
        G = nx.DiGraph()

        # Add all nodes first
        if debug:
            print("\n5. Adding nodes to graph...")
        for behavior in mean_probs.index:
            prob = behavior_probs.get(behavior, 0)
            if debug:
                print(f"Adding node: {behavior} with probability {prob}")
            G.add_node(behavior, probability=prob)

        # Add edges
        if debug:
            print("\n6. Adding edges (transitions) to graph...")
        edge_count = 0
        for from_behavior in mean_probs.index:
            for to_behavior in mean_probs.columns:
                prob = mean_probs.loc[from_behavior, to_behavior]
                if prob > threshold:
                    if debug:
                        print(f"Adding edge: {from_behavior} -> {to_behavior} (prob: {prob:.3f})")
                    G.add_edge(from_behavior, to_behavior,
                              weight=prob,
                              count=total_counts.loc[from_behavior, to_behavior])
                    edge_count += 1

        print("\n=== Final Graph Statistics ===")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print("Nodes:", list(G.nodes()))
        print("Edges:", list(G.edges()))

        return G

    def create_legend(self, palette, edge_scale=5, node_size=0.3, figsize=(8, 6)):
        dot_legend = graphviz.Digraph(comment='Legend')
        dot_legend.attr(rankdir='TB')  # Top to bottom layout

        # Create two separate clusters for behaviors and transition types
        with dot_legend.subgraph(name='cluster_behaviors') as behaviors:
            behaviors.attr(label='Behaviors', style='rounded', color='black')

            # Collect all unique behaviors and their colors
            G = self.create_markov_graph()
            n_nodes = len(G.nodes())
            node_colors = dict(zip(sorted(G.nodes()), palette))

            # Add each behavior node to legend with external label
            for node, color in node_colors.items():
                # Text color based on target
                if 'B' in node or 'BLDL' in node:
                    fontcolor = '#0A06CC'
                    desc = f'{node} (Blue male target)'
                elif 'Y' in node or 'YLDR' in node:
                    fontcolor = '#FA990A'
                    desc = f'{node} (Yellow male target)'
                else:
                    fontcolor = 'black'
                    desc = node

                # Create node and label separately
                behaviors.node(f'legend_{node}',
                             '',  # Empty label inside node
                             shape='circle',
                             style='filled',
                             width=str(node_size),
                             fillcolor=color,
                             fontcolor=fontcolor)

                # Add invisible edge to create spacing
                behaviors.edge(f'legend_{node}', f'label_{node}',
                             style='invis')

                # Add external label
                behaviors.node(f'label_{node}',
                             desc,
                             shape='plaintext',
                             fontcolor=fontcolor)

        with dot_legend.subgraph(name='cluster_transitions') as transitions:
            transitions.attr(label='Transition Types', style='rounded', color='black')

            # Add transition arrow examples
            # Create small nodes for the edges
            for i, (color, label) in enumerate([
                ('#0A06CC', 'Transition to Blue Male'),
                ('#FA990A', 'Transition to Yellow Male'),
                ('gray', 'Other Transitions')
            ]):
                # Create nodes for the edge
                node1 = f'trans_{i}_1'
                node2 = f'trans_{i}_2'
                label_node = f'trans_label_{i}'

                transitions.node(node1, '', shape='point')
                transitions.node(node2, '', shape='point')

                # Create edge with label
                transitions.edge(node1, node2, '', color=color, penwidth=str(edge_scale))

                # Add label node
                transitions.node(label_node, label, shape='plaintext')

                # Add invisible edge for spacing
                transitions.edge(node2, label_node, style='invis')

        # Save and render legend
        dot_legend.render('markov_legend', format='png', cleanup=True)

        # Display legend in separate figure
        img_legend = plt.imread('markov_legend.png')
        fig_legend = plt.figure(figsize=figsize)
        plt.imshow(img_legend)
        plt.axis('off')
        return fig_legend

    def plot_markov_graph(self, palette, group=None, threshold=0.1, figsize=(16, 16),
                         show_weights=True, edge_scale=5, node_scale=(1, 5)):
        """
        Plot the Markov model with enhanced visual features
        """
        G = self.create_markov_graph(group, threshold)

        # Create a dot graph with cluster support
        dot = graphviz.Digraph(comment='Behavioral Markov Model')
        dot.attr(rankdir='LR')

        # Create mapping dictionaries
        behavior_to_class = dict(zip(
            self.behavior_mappings['ReplaceWith'],
            self.behavior_mappings['Classification']
        ))

        # Get node probabilities for sizing
        probs = nx.get_node_attributes(G, 'probability')
        max_prob = max(probs.values()) if probs else 1
        min_size, max_size = node_scale

        # Generate distinct colors for each node using a qualitative colormap
        n_nodes = len(G.nodes())
        palette = sns.color_palette("muted", n_nodes).as_hex()
        # shuffle the palette to avoid adjacent nodes having similar colors
        node_colors = dict(zip(G.nodes(), palette))

        # Create subgraphs by classification
        classifications = set(behavior_to_class.values())
        for classification in classifications:
            with dot.subgraph(name=f'cluster_{classification}') as c:
                c.attr(label=classification.title(), style='rounded', color='gray')

                for node in G.nodes():
                    node_class = behavior_to_class.get(node, None)
                    if node_class == classification:
                        size = min_size + (probs.get(node, 0) / max_prob) * (max_size - min_size)

                        # Unique color for each node
                        fillcolor = node_colors[node]

                        # Text color based on target
                        if 'B' in node or 'BLDL' in node:
                            fontcolor = '#0A06CC'
                        elif 'Y' in node or 'YLDR' in node:
                            fontcolor = '#FA990A'
                        else:
                            fontcolor = 'black'

                        c.node(node,
                              shape='circle',
                              style='filled',
                              fillcolor=fillcolor,
                              fontcolor=fontcolor,
                              width=str(size))

        # Add edges with enhanced styling
        for u, v in G.edges():
            weight = G[u][v]['weight']
            penwidth = edge_scale * weight

            # Edge color based on target behavior
            if 'B' in v or 'BLDL' in v:
                edge_color = '#0A06CC'
            elif 'Y' in v or 'YLDR' in v:
                edge_color = '#FA990A'
            else:
                edge_color = 'gray'

            # Edge label with probability percentage
            label = f'{weight:.1%}' if show_weights else ''

            dot.edge(u, v,
                    penwidth=str(penwidth),
                    color=edge_color,
                    label=label)

        # Save and render
        dot.render('markov_model', format='png', cleanup=True)

        # Display using matplotlib
        img = plt.imread('markov_model.png')
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        return plt.gcf()
# ext = BehaviorExtractor()
# =============================================================================
# Retrieve data by group and/or file
# =============================================================================
# ext.extract_folder('data')
# group, file = ext.get_file_at_index(0)
# -----------------------------------------------------------------------------
# Analyze individual files
# -----------------------------------------------------------------------------
# print(ext.analyze_time_in_states(group, file))
# print(ext.analyze_behavior_probabilities(group, file))
# print(ext.analyze_transitions(group, file))
# -----------------------------------------------------------------------------
# Analyze by group
# -----------------------------------------------------------------------------
# print(ext.analyze_time_in_states(group))
# print(ext.analyze_behavior_probabilities(group))
# print(ext.analyze_transitions(group))
# -----------------------------------------------------------------------------
# Analyze all data
# -----------------------------------------------------------------------------
# print(ext.analyze_time_in_states())
# print(ext.analyze_behavior_probabilities())
# print(ext.analyze_transitions())
# =============================================================================
# Export all data to CSV
# ext.export_analyses('output')
# =============================================================================
# Get summary statistics for a specific group
# blue_stats = ext.get_behavior_summary_stats(group='blue')
# print(f"Average time in 'attack blue (left)': {blue_stats['mean']['attack blue (left)']:.2f} seconds")
# print(f"Standard deviation: {blue_stats['std']['attack blue (left)']:.2f} seconds")

# Get summary statistics across all groups
# all_stats = ext.get_behavior_summary_stats()

# Compare statistics across groups
# group_comparisons = ext.compare_groups_stats()

# Export all summary statistics
# ext.export_summary_stats('summary_stats_output')
# Get summary statistics for transitions in a specific group
# blue_trans_stats = ext.get_transition_summary_stats(group='blue')

# Print formatted summary
# ext.print_transition_summary('blue')  # For one group
# ext.print_transition_summary()        # For all groups

# Export all summary statistics
# ext.export_transition_summary_stats('transition_stats_output')
# =============================================================================
# Get classification statistics
# ext.print_classification_summary('blue')
# ext.verify_mappings()
# ext.plot_markov_model(group= 'blue', threshold=0.00)
# plt.show()
# Create enhanced visualization
# Assuming you have already instantiated your BehaviorExtractor as 'extractor'
# and loaded your data

# Create the Markov visualization
# Optional: Save the figure
#fig_graph = ext.plot_markov_graph(palette, threshold=0.10,
#                                  edge_scale=20,
#                                  node_scale=(1, 1),
#                                  show_weights=True
#                                  )
# save the figure
#fig_graph.savefig('markov_model.png', dpi=300)

#fig_legend = ext.create_legend(palette, edge_scale=3.)
#fig_legend.savefig('markov_legend.png', dpi=300)

#plt.show()
