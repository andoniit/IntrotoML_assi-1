import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, label=None, attribute=None, entropy=None, split_condition=None):
        self.label = label
        self.attribute = attribute
        self.entropy = entropy
        self.split_condition = split_condition
        self.children = {}

def entropy(y, class_list):
    unique_labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def calculate_total_entropy(dataset, label, class_list):
    return entropy(dataset[label], class_list)

def calculate_entropy(data_subset, label, class_list):
    return entropy(data_subset[label], class_list)

def calculate_information_gain(feature_name, dataset, label, class_list):
    total_rows = dataset.shape[0]
    total_entropy = calculate_total_entropy(dataset, label, class_list)
    information_gain = 0.0

    for feature_value in dataset[feature_name].unique():
        feature_value_data = dataset[dataset[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_probability = feature_value_count / total_rows

        if feature_value_probability > 0:
            feature_value_entropy = calculate_entropy(feature_value_data, label, class_list)
            information_gain += (feature_value_count / total_rows) * feature_value_entropy

    return total_entropy - information_gain

def my_ID3(X, y, attributes, class_list):
    if len(np.unique(y)) == 1:
        return Node(label=str(y.iloc[0]))

    if len(attributes) == 0:
        majority_label = str(y.mode().iloc[0])
        return Node(label=majority_label)

    max_gain_attribute = max(attributes, key=lambda attr: calculate_information_gain(attr, X, 'Output', class_list))
    max_gain_value = calculate_information_gain(max_gain_attribute, X, 'Output', class_list)
    max_gain_entropy = calculate_total_entropy(X, 'Output', class_list)

    print(f"Attribute {max_gain_attribute} with Gain = {max_gain_value:.4f} and Entropy = {max_gain_entropy:.4f} is chosen as the decision attribute.")

    root = Node(attribute=max_gain_attribute, entropy=max_gain_entropy)

    for value in X[max_gain_attribute].unique():
        subset_X = X[X[max_gain_attribute] == value].drop(columns=[max_gain_attribute])
        subset_y = y[X[max_gain_attribute] == value]
        split_condition = f"{max_gain_attribute} = {value}"

        if subset_X.empty:
            majority_label = str(y.mode().iloc[0])
            root.children[value] = Node(label=majority_label, split_condition=split_condition)
        else:
            root.children[value] = my_ID3(subset_X, subset_y, attributes - {max_gain_attribute}, class_list)

    return root

def visualize_tree(node, parent_name=None, text_spacing=(0.5, 0.1)):
    if node.label is not None:
        label = f"{node.label}\nEntropy: {node.entropy:.4f}" if node.entropy is not None else node.label
        plt.text(*text_spacing, label, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    else:
        label = f"{node.attribute}\nEntropy: {node.entropy:.4f}" if node.entropy is not None else node.attribute
        plt.text(*text_spacing, label, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    if parent_name is not None:
        plt.plot([parent_name[0], text_spacing[0]], [parent_name[1], text_spacing[1]], 'k-')

    for value, child_node in node.children.items():
        new_text_spacing = (text_spacing[0] + (value - 0.5) * 0.2, text_spacing[1] - 0.2)
        visualize_tree(child_node, text_spacing, new_text_spacing)

# Read the cleaned dataset
file_path = 'lab01_dataset_1_updated.csv'
df = pd.read_csv(file_path)

# Select attributes (excluding the output column)
attributes = set(df.columns[:-1])

# Build the decision tree
decision_tree = my_ID3(df, df['Output'], attributes, df['Output'].unique())

# Set up the plot
plt.figure(figsize=(10, 6))
plt.axis('off')

# Visualize the decision tree
visualize_tree(decision_tree)

# Show the plot
plt.show()
