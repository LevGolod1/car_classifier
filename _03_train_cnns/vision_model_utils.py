from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np

def custom_classification_report(true_labels, predicted_labels, class_names):
    '''

              precision    recall  f1-score   support

       sedan       0.71      0.70      0.71       499
         suv       0.49      0.26      0.34       210
       truck       0.51      0.67      0.58       328

    accuracy                           0.60      1037
   macro avg       0.57      0.54      0.54      1037
weighted avg       0.60      0.60      0.59      1037
    '''

  # Get precision, recall, f1-score, and support for each class
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, labels=class_names, average=None)

    # Compute overall accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Compute macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    # Create a pandas DataFrame for clean formatting
    report_df = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': support
    }, index=class_names)

    # Add the averages and accuracy to the report
    avg_stats = pd.DataFrame({
        'precision': [macro_precision, weighted_precision],
        'recall': [macro_recall, weighted_recall],
        'f1-score': [macro_f1, weighted_f1],
        'support': [np.sum(support), np.sum(support)]  # Total support for both avg rows
    }, index=['macro avg', 'weighted avg'])

    # Add the accuracy row
    accuracy_row = pd.DataFrame({
        'precision': [None],
        'recall': [None],
        'f1-score': [accuracy],
        'support': [len(true_labels)]
    }, index=['accuracy'])

    # Combine the individual class report, accuracy, and average rows
    report_df = pd.concat([report_df, accuracy_row, avg_stats])

    # Format the DataFrame to match the Terraform report style
    report_str = report_df.to_string(formatters={
        'precision': '{:0.2f}'.format,
        'recall': '{:0.2f}'.format,
        'f1-score': '{:0.2f}'.format,
        'support': '{:0.0f}'.format
    })
    report_str=report_str.replace('NaN','   ')
    return report_df, report_str