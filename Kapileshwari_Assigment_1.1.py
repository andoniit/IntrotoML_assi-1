# Name-Anirudha Kapileshwari
# Spring 2024 Introduction to Machine Learning (CS-484-01)
# prof Shouvik Roy

# Assignment 1 CS 484
# Decision Tree Learning

# Inspect the dataset titled lab01_dataset_1.csv which has a mixture of numerical and categorical data. 
# Your task will be to write a function my_ID3( ) which can create a decision tree for the given dataset using the ID3 algorithm. However,
# before doing that, you will be have to perform some data processing tasks. Here are all the required tasks in order 


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Node:
    def __init__(self, label=None, attribute=None, entropy=None, information_gain=None, split_condition=None):
        self.label = label
        self.attribute = attribute
        self.entropy = entropy
        self.information_gain = information_gain
        self.split_condition = split_condition
        self.children = {}

def entropy(y, class_list):
    unique_labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def total_entropy(dataset, label, class_list):
    return entropy(dataset[label], class_list)

def calculate_entropy(data_subset, label, class_list):
    return entropy(data_subset[label], class_list)

def information_gain(feature_name, dataset, label, class_list):
    total_rows = dataset.shape[0]
    total_ent = total_entropy(dataset, label, class_list)  # Rename the variable here
    information_gain = 0.0

    for feature_value in dataset[feature_name].unique():
        feature_value_data = dataset[dataset[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_probability = feature_value_count / total_rows

        if feature_value_probability > 0:
            feature_value_entropy = calculate_entropy(feature_value_data, label, class_list)
            information_gain += (feature_value_count / total_rows) * feature_value_entropy

    return total_ent - information_gain


def my_ID3(X, y, attributes, class_list):
    if len(np.unique(y)) == 1:
        return Node(label=str(y.iloc[0]))

    if len(attributes) == 0:
        majority_label = str(y.mode().iloc[0])
        return Node(label=majority_label)

    max_gain_attribute = max(attributes, key=lambda attr: information_gain(attr, X, 'Output', class_list))
    max_gain_value = information_gain(max_gain_attribute, X, 'Output', class_list)
    max_gain_entropy = total_entropy(X, 'Output', class_list)

    print(f"Attribute {max_gain_attribute} with Gain = {max_gain_value:.4f} and Entropy = {max_gain_entropy:.4f} is chosen as the decision attribute.")

    root = Node(attribute=max_gain_attribute, entropy=max_gain_entropy, information_gain=max_gain_value)

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

def graphical_tree(node, parent_name=None, text_spacing=(0.5, 0.1)):
    if node.label is not None:
        label = f"{node.label}\nEntropy: {node.entropy:.4f}\nGain: {node.information_gain:.4f}" if node.entropy is not None else node.label
        plt.text(*text_spacing, label, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    else:
        label = f"{node.attribute}\nEntropy: {node.entropy:.4f}\nGain: {node.information_gain:.4f}" if node.entropy is not None else node.attribute
        plt.text(*text_spacing, label, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    if parent_name is not None:
        plt.plot([parent_name[0], text_spacing[0]], [parent_name[1], text_spacing[1]], 'k-')

    for value, child_node in node.children.items():
        new_text_spacing = (text_spacing[0] + (value - 0.5) * 0.2, text_spacing[1] - 0.2)
        graphical_tree(child_node, text_spacing, new_text_spacing)


# This function is to update the data accordingly
def update_data():

    file_path = 'lab01_dataset_1.csv'
    df = pd.read_csv(file_path)

    df.sort_values(by='Score', inplace=True)

    threshold_46 = 46.0
    threshold_69_5 = 69.5
    threshold_81_5 = 81.5

    df['Score_46.0'] = df['Score'] < threshold_46
    df['Score_69.5'] = df['Score'] < threshold_69_5
    df['Score_81.5'] = df['Score'] < threshold_81_5

    # 1) ID3 cannot handle continuous numerical data. Perform necessary operations to handle all continuous-valued
    # attributes. Do not forget to show the output i.e., the updated dataset after handling continuous-valued attributes. 
    df[['Score_46.0', 'Score_69.5', 'Score_81.5']] = df[['Score_46.0', 'Score_69.5', 'Score_81.5']].astype(bool)

    df = df.drop('Score', axis=1)

    
    column_order = [col for col in df.columns if col != 'Output'] + ['Output']
    df = df[column_order]

    # 2-a)Check if the dataset has any missing values.
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Dataset has {missing_values} missing values.")
        # You can choose to handle missing values as needed.

    # 2-b)Check if the dataset has any redundant or repeated input sample.
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        print("Dataset has redundant or repeated input samples.")

        
        print("Duplicate rows:")
        print(duplicate_rows)

        
        df = df.drop_duplicates()
        print("Duplicate rows removed.")

    # 2-c)Check if the dataset has any contradicting <input, output> pairs. 
    contradicting_rows = df[df.duplicated(subset=['Mood', 'Effort', 'Score_46.0', 'Score_69.5', 'Score_81.5', 'Output'], keep=False)]
    if not contradicting_rows.empty:
        print("Dataset has contradicting <input, output> pairs.")
        df = df.drop_duplicates(subset=['Mood', 'Effort', 'Score_46.0', 'Score_69.5', 'Score_81.5', 'Output'])
        print("Contradicting rows removed.")

    output_file_path = 'lab01_dataset_1_updated.csv'
    df.to_csv(output_file_path, index=False)

    print(f"Modified and cleaned dataset saved to {output_file_path}.")


#Your function my_ID3( ) should operate in a manner such that after ever round of decision making, 
#it will output the attributes and its associated gain, with a message stating “Attribute X with Gain = Y is chosen as 
#the decision attribute”. Once your function completes, it should output the decision tree. The representation of the 
#decision tree is upto you. You can choose either a textual representation or a graphical one; either is fine.
def run_id3():
    file_path = 'lab01_dataset_1_updated.csv'
    df = pd.read_csv(file_path)
    attributes = set(df.columns[:-1])
    decision_tree = my_ID3(df, df['Output'], attributes, df['Output'].unique())

    plt.figure(figsize=(10, 6))
    plt.axis('off')
    graphical_tree(decision_tree)
    plt.show()

def main():
    exit_flag = False

    while not exit_flag:
       
        choice = input("Press '1' to update data, '2' to run the my_ID3, or 'q' to quit: ").lower()

        if choice == '1':
            update_data()
        elif choice == '2':
            run_id3()
        elif choice == 'q':
            print("Quitting the program. Goodbye!")
            exit_flag = True
        else:
            print("Invalid choice. Please enter '1', '2', or 'q'.")

if __name__ == "__main__":
    main()
