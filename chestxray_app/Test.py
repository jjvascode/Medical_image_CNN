#Import libraries
import pandas as pd
import ast

#Used to map test_label.csv indices to label names and save the results to a new csv
def format():
    #Define labels
    labels = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    #Load test labels csv
    df = pd.read_csv('test_label.csv')
    #Convert each label entry string to a list with a single index entry
    df['labels'] = df['labels'].apply(ast.literal_eval)
    #Map each index entry to its corresponding label name to create an array of label names for each entry
    df['labels'] = df['labels'].apply(lambda idxs: [labels[i] for i in idxs])
    #Convert the array of label names to a single string with each label name separated by commas
    df['labels'] = df['labels'].apply(lambda x: ', '.join(x))
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
    merged_df['is_match'] = merged_df.apply(lambda row: row['labels'] == row['prediction'], axis=1)
    #Evaulate accuracy based on the number of labels and predictions matches divided by the number image_filname matches
    total = len(merged_df)
    matches = merged_df['is_match'].sum()
    #else 0 to prevent divde by 0 error
    accuracy = matches / total * 100 if total > 0 else 0
    #Print the number of filename matches, prediction matches, and accuracy
    print(f'Filename Matches: {total}')
    print(f'Prediction Matches: {matches}')
    print(f'Accuracy: {accuracy:.2f}%')
    #Save evaluation results
    merged_df.to_csv('evaluation.csv', index=False)

    return

format()
test('mapped_label.csv', 'predictions.csv')