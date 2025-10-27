import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TwitterDataVisualizer:
    """Comprehensive visualization module for Twitter rumor detection datasets"""
    
    def __init__(self, dataset_path: str, dataset_name: str = "twitter15"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        
    def load_dataset_for_visualization(self):
        """Load dataset for visualization purposes"""
        from .data_preprocessor import TwitterDatasetPreprocessor
        
        preprocessor = TwitterDatasetPreprocessor(self.dataset_path, self.dataset_name)
        return preprocessor
    
    def plot_label_distribution(self, save_path: str = None):
        """Plot label distribution with enhanced styling"""
        preprocessor = self.load_dataset_for_visualization()
        stats = preprocessor.get_dataset_stats()
        
        labels = list(stats['label_distribution'].keys())
        counts = list(stats['label_distribution'].values())
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(labels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title(f'Label Distribution - {self.dataset_name.upper()}', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Rumor Labels', fontsize=12)
        ax1.set_ylabel('Number of Tweets', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax2.pie(counts, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title(f'Label Proportions - {self.dataset_name.upper()}', fontsize=16, fontweight='bold')
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return stats['label_distribution']
    
    def plot_feature_distributions(self, save_path: str = None):
        """Plot distributions of key features"""
        preprocessor = self.load_dataset_for_visualization()
        stats = preprocessor.get_dataset_stats()
        
        # Extract feature data
        all_sentiments = []
        all_word_counts = []
        all_hashtag_counts = []
        all_mention_counts = []
        
        for tweet_id, text in preprocessor.source_tweets.items():
            features = preprocessor.extract_features(text)
            all_sentiments.append(features['sentiment'])
            all_word_counts.append(features['word_count'])
            all_hashtag_counts.append(features['hashtag_count'])
            all_mention_counts.append(features['mention_count'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sentiment distribution
        ax1.hist(all_sentiments, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sentiment Score (-1 to 1)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.axvline(np.mean(all_sentiments), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(all_sentiments):.3f}')
        ax1.legend()
        
        # Word count distribution
        ax2.hist(all_word_counts, bins=30, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax2.set_title('Word Count Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Words', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.axvline(np.mean(all_word_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_word_counts):.1f}')
        ax2.legend()
        
        # Hashtag count distribution
        ax3.hist(all_hashtag_counts, bins=20, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax3.set_title('Hashtag Count Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Hashtags', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.axvline(np.mean(all_hashtag_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_hashtag_counts):.2f}')
        ax3.legend()
        
        # Mention count distribution
        ax4.hist(all_mention_counts, bins=20, color='#96CEB4', alpha=0.7, edgecolor='black')
        ax4.set_title('Mention Count Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Mentions', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.axvline(np.mean(all_mention_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_mention_counts):.2f}')
        ax4.legend()
        
        plt.suptitle(f'Feature Distributions - {self.dataset_name.upper()}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'sentiment': all_sentiments,
            'word_count': all_word_counts,
            'hashtag_count': all_hashtag_counts,
            'mention_count': all_mention_counts
        }
    
    def plot_feature_correlations(self, save_path: str = None):
        """Plot correlation matrix of numerical features"""
        preprocessor = self.load_dataset_for_visualization()
        
        # Extract features for correlation analysis
        feature_data = []
        for tweet_id, text in preprocessor.source_tweets.items():
            features = preprocessor.extract_features(text)
            feature_data.append([
                features['sentiment'],
                features['subjectivity'],
                features['word_count'],
                features['char_count'],
                features['hashtag_count'],
                features['mention_count'],
                features['url_count'],
                features['exclamation_count'],
                features['question_count'],
                features['emoji_count'],
                features['avg_word_length'],
                features['capital_ratio']
            ])
        
        # Create DataFrame
        feature_names = ['Sentiment', 'Subjectivity', 'Word_Count', 'Char_Count', 
                        'Hashtag_Count', 'Mention_Count', 'URL_Count', 
                        'Exclamation_Count', 'Question_Count', 'Emoji_Count',
                        'Avg_Word_Length', 'Capital_Ratio']
        
        df = pd.DataFrame(feature_data, columns=feature_names)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(f'Feature Correlation Matrix - {self.dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return corr_matrix
    
    def plot_temporal_analysis(self, save_path: str = None):
        """Plot temporal analysis of tweets"""
        preprocessor = self.load_dataset_for_visualization()
        
        # Extract temporal features
        hours = []
        days = []
        months = []
        
        for tweet_id in preprocessor.labels.keys():
            if tweet_id in preprocessor.source_tweets:
                temporal_features = preprocessor.create_temporal_features(tweet_id)
                hours.append(temporal_features['hour_of_day'])
                days.append(temporal_features['day_of_week'])
                months.append(temporal_features['month'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Hour of day distribution
        hour_counts = Counter(hours)
        hour_values = [hour_counts.get(h, 0) for h in range(24)]
        
        ax1.bar(range(24), hour_values, color='#FF6B6B', alpha=0.7)
        ax1.set_title('Tweet Activity by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Tweets', fontsize=12)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
        
        # Day of week distribution
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = Counter(days)
        day_values = [day_counts.get(d, 0) for d in range(7)]
        
        ax2.bar(range(7), day_values, color='#4ECDC4', alpha=0.7)
        ax2.set_title('Tweet Activity by Day of Week', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Number of Tweets', fontsize=12)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(day_names, rotation=45)
        
        # Month distribution
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_counts = Counter(months)
        month_values = [month_counts.get(m, 0) for m in range(1, 13)]
        
        ax3.bar(range(1, 13), month_values, color='#45B7D1', alpha=0.7)
        ax3.set_title('Tweet Activity by Month', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month', fontsize=12)
        ax3.set_ylabel('Number of Tweets', fontsize=12)
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(month_names)
        
        # Weekend vs Weekday
        weekend_count = sum(1 for d in days if d >= 5)
        weekday_count = len(days) - weekend_count
        
        ax4.pie([weekday_count, weekend_count], labels=['Weekday', 'Weekend'], 
                autopct='%1.1f%%', colors=['#96CEB4', '#FFEAA7'], startangle=90)
        ax4.set_title('Weekday vs Weekend Distribution', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Temporal Analysis - {self.dataset_name.upper()}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'hours': hours,
            'days': days,
            'months': months,
            'weekend_ratio': weekend_count / len(days)
        }
    
    def plot_propagation_graph_example(self, save_path: str = None):
        """Plot an example propagation graph"""
        preprocessor = self.load_dataset_for_visualization()
        
        # Get first tweet as example
        tweet_id = list(preprocessor.labels.keys())[0]
        tweet_text = preprocessor.source_tweets[tweet_id]
        features = preprocessor.extract_features(tweet_text)
        temporal_features = preprocessor.create_temporal_features(tweet_id)
        
        # Build propagation graph
        G = preprocessor.build_propagation_graph(tweet_id, features, temporal_features)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=[n for n in G.nodes() if G.nodes[n]['type'] == 'source'],
                              node_color='#FF6B6B', node_size=1000, alpha=0.8, label='Source Tweet')
        
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=[n for n in G.nodes() if G.nodes[n]['type'] == 'reply'],
                              node_color='#4ECDC4', node_size=600, alpha=0.8, label='Replies')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels
        labels = {}
        for node in G.nodes():
            if G.nodes[node]['type'] == 'source':
                labels[node] = f"Source\n{G.nodes[node]['word_count']} words\n{G.nodes[node]['sentiment']:.2f}"
            else:
                labels[node] = f"Reply\n{G.nodes[node]['word_count']} words\n{G.nodes[node]['sentiment']:.2f}"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title(f'Example Propagation Graph - {self.dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
        plt.legend()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return G
    
    def create_comprehensive_report(self, save_dir: str = "outputs/visualizations"):
        """Create comprehensive visualization report"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Creating comprehensive visualization report for {self.dataset_name}...")
        
        # Generate all visualizations
        self.plot_label_distribution(f"{save_dir}/label_distribution_{self.dataset_name}.png")
        self.plot_feature_distributions(f"{save_dir}/feature_distributions_{self.dataset_name}.png")
        self.plot_feature_correlations(f"{save_dir}/feature_correlations_{self.dataset_name}.png")
        self.plot_temporal_analysis(f"{save_dir}/temporal_analysis_{self.dataset_name}.png")
        self.plot_propagation_graph_example(f"{save_dir}/propagation_graph_{self.dataset_name}.png")
        
        print(f"All visualizations saved to {save_dir}/")
        
        # Create summary statistics
        preprocessor = self.load_dataset_for_visualization()
        stats = preprocessor.get_dataset_stats()
        
        print(f"\n📊 DATASET SUMMARY - {self.dataset_name.upper()}")
        print("=" * 50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Label distribution: {stats['label_distribution']}")
        
        print(f"\nFeature Statistics:")
        for feature, values in stats['feature_stats'].items():
            print(f"  {feature}: mean={values['mean']:.3f}, std={values['std']:.3f}")
        
        return stats

def main():
    """Main function to run all visualizations"""
    dataset_path = "data/Rumor Detection Dataset (Twitter15 and Twitter16)"
    
    # Create visualizer for Twitter15
    print("🚀 Creating visualizations for Twitter15 dataset...")
    visualizer_15 = TwitterDataVisualizer(dataset_path, "twitter15")
    stats_15 = visualizer_15.create_comprehensive_report("outputs/visualizations_twitter15")
    
    # Create visualizer for Twitter16
    print("\n🚀 Creating visualizations for Twitter16 dataset...")
    visualizer_16 = TwitterDataVisualizer(dataset_path, "twitter16")
    stats_16 = visualizer_16.create_comprehensive_report("outputs/visualizations_twitter16")
    
    # Compare datasets
    print(f"\n📈 DATASET COMPARISON")
    print("=" * 50)
    print(f"Twitter15: {stats_15['total_samples']} samples")
    print(f"Twitter16: {stats_16['total_samples']} samples")
    
    print(f"\nLabel distribution comparison:")
    for label in set(list(stats_15['label_distribution'].keys()) + list(stats_16['label_distribution'].keys())):
        count_15 = stats_15['label_distribution'].get(label, 0)
        count_16 = stats_16['label_distribution'].get(label, 0)
        print(f"  {label}: Twitter15={count_15}, Twitter16={count_16}")

if __name__ == "__main__":
    main()
