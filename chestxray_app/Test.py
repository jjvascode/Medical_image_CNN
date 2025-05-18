#Import libraries
import pandas as pd
import ast

#Used to map test_label.csv indices to label names and save the results to a new csv
def format():
    #Define labels
    labels = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    binary_labels = ['No Finding', 'Finding']
    #Load test labels csv
    df = pd.read_csv('test_labels.csv')
    #Convert each label entry string to a list with a single index entry
    df['labels'] = df['labels'].apply(ast.literal_eval)
    #Remove all entries with more than 1 label
    df = df[df['labels'].apply(lambda x: len(x) == 1)]
    #Map first and only index entry to its corresponding label name to create a label name for each entry
    df['labels'] = df['labels'].apply(lambda idxs: labels[idxs[0]])
    df['binary_label'] = df['binary_label'].apply(lambda x: binary_labels[int(x)])
    #Save the new csv
    df.to_csv('mapped_label.csv', index=False)
    return

#Used to evalulate predictions made with mapped labels
def test(test_csv, prediction_csv):
    #Load mapped labels csv and prediction csv into dataframes
    test_df = pd.read_csv(test_csv)
    prediction_df = pd.read_csv(prediction_csv)
    #Merge dataframes on image_filename column
    merged_df = pd.merge(test_df, prediction_df, on='image_filename', suffixes=('_true', '_pred'))
    #Create a new column that's either true or false depend on if the labels and prediction entry on a given row match
    merged_df['illness_match'] = merged_df.apply(lambda row: row['labels'] == row['illness prediction'], axis=1)
    merged_df['binary_match'] = merged_df.apply(lambda row: row['binary_label'] == row['binary prediction'], axis=1)
    #Evaulate accuracy based on the number of labels and predictions matches divided by the number image_filname matches
    total = len(merged_df)
    illness_matches = merged_df['illness_match'].sum()
    binary_matches = merged_df['binary_match'].sum()
    #else 0 to prevent divde by 0 error
    illness_accuracy = illness_matches / total * 100 if total > 0 else 0
    binary_accuracy = binary_matches / total * 100 if total > 0 else 0
    #Print the number of filename matches, prediction matches, and accuracy
    print(f'Filename Matches: {total}')
    print(f'Illness Matches: {illness_matches}')
    print(f'Illness Accuracy: {illness_accuracy:.2f}%')
    print(f'Binary Matches: {binary_matches}')
    print(f'Binary Accuracy: {binary_accuracy:.2f}%')
    #Save evaluation results
    merged_df.to_csv('evaluation.csv', index=False)

    return

format()
test('mapped_label.csv', 'predictions.csv')