""" 
The present module is a complete machine learning pipeline for the MNIST dataset. 
It provides functions for data extraction, data transformation, model training, 
model evaluation, and saving trained models and metrics objects. For an interpretation 
of the pipeline results, please visit the repository's README.md file.
"""
 
import sys
import time
import random
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def unfold_mnist_data():
    # Load MNIST dataset using Keras
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshape training and test data to a flat array of 784 elements
    X_train = tf.reshape(X_train, (X_train.shape[0], -1))
    X_test = tf.reshape(X_test, (X_test.shape[0], -1))
    # Return the unfolded training and test data along with their corresponding labels
    return X_train, y_train, X_test, y_test


def plot_class_counts(y_train, y_test):
    # Combine training and test labels
    y_mnist = np.hstack((y_train, y_test))

    # Create a figure with three subplots
    plt.figure(figsize=(4, 9))

    # Plot the overall class distribution for the entire MNIST dataset
    plt.subplot(3, 1, 1)
    plt.bar_label(sns.countplot(x=y_mnist, palette=['grey'], width=.9).containers[0], size=6)
    plt.title(f'MNIST class counts ({len(y_mnist)} samples)', size=9, pad=1.5)
    plt.xlabel('Class', labelpad=0, size=8)
    plt.ylabel('Count', labelpad=1, size=8)
    plt.xlim(-0.6, 9.6)
    plt.ylim(0, 8500)
    plt.xticks(size=6)
    plt.yticks(np.arange(1000, 8001, 1000), size=6)
    plt.tick_params(axis='both', pad=1, length=2, direction='inout')

    # Plot the class distribution for the training set
    plt.subplot(3, 1, 2)
    plt.bar_label(sns.countplot(x=y_train, palette=['grey'], width=.9).containers[0], size=6)
    plt.title(f"MNIST 'train set' class counts ({len(y_train)} samples)", size=9, pad=1.5)
    plt.xlabel('Class', labelpad=0, size=8)
    plt.ylabel('Count', labelpad=1, size=8)
    plt.xlim(-0.6, 9.6)
    plt.ylim(0, 7300)
    plt.xticks(size=6)
    plt.yticks(np.arange(1000, 7301, 1000), size=6)
    plt.tick_params(axis='both', pad=1, length=2, direction='inout')

    # Plot the class distribution for the test set
    plt.subplot(3, 1, 3)
    plt.bar_label(sns.countplot(x=y_test, palette=['grey'], width=.9).containers[0], size=6)
    plt.title(f"MNIST 'test set' class counts ({len(y_test)} samples)", size=9, pad=1.5)
    plt.xlabel('Class', labelpad=0, size=8)
    plt.ylabel('Count', labelpad=0, size=8)
    plt.xlim(-0.6, 9.6)
    plt.ylim(0, 1300)
    plt.xticks(size=6)
    plt.yticks(np.arange(200, 1201, 200), size=6)
    plt.tick_params(axis='both', pad=1, length=2, direction='inout')

    # Adjust the layout of the subplots and save the plot as an image file
    plt.tight_layout()
    directory = './Evaluations'
    filename = 'Class_Counts.png'
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{directory}/{filename}')
    plt.show()

    # Print a message indicating where the plot has been saved
    print(f'Class count plots saved in ~{directory} directory.\n')


def get_model_choice():
    # grab the user's choice of model(s) to train, return a string list of model choices.
    model_choices = [
        "LR",
        "RFC",
        "SVC",
        "KNC",
        "LR, RFC",
        "LR, SVC",
        "LR, KNC",
        "RFC, SVC",
        "RFC, KNC",
        "SVC, KNC",
        "LR, RFC, SVC",
        "LR, RFC, KNC",
        "LR, SVC, KNC",
        "RFC, SVC, KNC",
        "LR, RFC, SVC, KNC"]

    # Print the available model choices to the user
    print("Please select the model(s) you would like to train:\n")
    for i, model_choice in enumerate(model_choices):
        print(f'{i+1}. {model_choice}')

    # Prompt the user to enter their choice and return it as a list of strings
    while True:
        try:
            choice = int(input("\nEnter your choice: ")) - 1
            if choice < 0 or choice >= len(model_choices):
                raise ValueError
            return model_choices[choice].split(', ')
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(model_choices)}.")


def scale_mnist_features(X_train, X_test):
    # initialize a StandardScaler object
    scaler = StandardScaler()
    # fit the scaler to the training data and transform it
    X_train = scaler.fit_transform(X_train)
    # transform the test data using the scaler fit to the training data
    X_test = scaler.transform(X_test)
    # return the scaled training and test data
    return X_train, X_test


def train_model_and_calc_training_time(model, X_train, y_train):
    # Get the current time before fitting the model
    start = time.time()
    # Fit the model using the provided X_train and y_train data
    model.fit(X_train, y_train)
    # Get the current time again after fitting the model
    end = time.time()
    # Calculate the time it took to fit the model
    fitting_time = end - start
    # Return the fitting time as the output of the function
    return np.round(fitting_time*1000, 4)


def predict_targets_and_calc_prediction_time(model, X_test):
    # start the timer to measure the inference time of the model
    start = time.time()
    # use the trained model to predict the target values for the test data
    predictions = model.predict(X_test)
    # stop the timer and calculate the time taken for inference
    end = time.time()
    # define latency as inference time spent over a single digit
    inference_time = (end - start) / len(X_test)
    # return the predicted values and the inference time
    return predictions, np.round(inference_time*1000, 4)


def display_and_save_metrics(model, y_test, preds):
    # Extract the name of the input model
    model_name = model.__class__.__name__
    # Calculate the classification report for the predicted labels
    cr = classification_report(y_test, preds, digits=3, output_dict=True)
    # Create a directory named 'Evaluations' (if it doesn't exist)
    directory = './Evaluations'
    Path(directory).mkdir(parents=True, exist_ok=True)
    # Save the classification report to a pickled file
    joblib.dump(cr, f'{directory}/{model_name}_Evaluations.pkl')
    # Print the CR to the console
    print(f'\n{model_name} model classification report:\n')
    cr_df = pd.DataFrame.from_dict(cr)
    print(np.transpose(cr_df))
    # Extract the class labels from the model
    classes = model.classes_
    # Calculate the confusion matrix for the test and predicted labels
    cm = confusion_matrix(y_test, preds, labels=classes)
    # Display the confusion matrix plot
    ConfusionMatrixDisplay(cm, display_labels=classes).plot()
    # Display the title of the confusion matrix plot
    plt.title(f'MNIST Digit Classification\nConfusion Matrix for {model_name}')
    # Save the confusion matrix plot as an image
    filename = f'ConfusionMatrix_{model_name}.png'
    plt.savefig(f'{directory}/{filename}')
    # Display the confusion matrix plot
    plt.show()
    # Print a message indicating where the metrics and plot are saved
    print(f'\nPlot, metrics saved in ~{directory} directory.')


def save_model(model):
    # Get the class name of the model
    model_name = model.__class__.__name__
    # Set the directory where the model will be saved
    directory = './Models'
    # Create the directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)
    # Save the model to a file in the specified directory
    joblib.dump(model, f'{directory}/{model_name}_mnist.joblib')
    # Print a message to indicate where the model was saved
    print(f'{model_name} model object saved in ~{directory} directory.')


def compare_display_save_final_evaluations(model_performances):
    directory = './Evaluations'
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Extract the names of the models from the columns of the DataFrame
    model_names = model_performances.columns
    # Create a name mapping dict for brevity of text displayed on the plot
    modelname_map = {'KNeighborsClassifier': 'KNC', 'LogisticRegression': 'LR',
                     'RandomForestClassifier': 'RFC', 'SVC': 'SVC'}
    model_names = model_names.map(modelname_map)
    training_durations = model_performances.iloc[0].values
    inference_durations = model_performances.iloc[1].values
    accuracy_scores = model_performances.iloc[2].values * 100

    # Draw barplots for each metric using Seaborn
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 3), dpi=130)

    plot = sns.barplot(x=model_names, y=training_durations, ax=axs[0], palette='viridis')
    axs[0].bar_label(plot.containers[0], size=5)
    axs[0].set_title('Model vs. Training Duration', fontsize=6, pad=3)
    axs[0].set_ylabel(ylabel='Time (ms.)', fontsize=5)
    axs[0].tick_params(axis='both', pad=2, length=2, direction='inout', labelsize=5)

    plot = sns.barplot(x=model_names, y=inference_durations, ax=axs[1], palette='viridis')
    axs[1].bar_label(plot.containers[0], size=5)
    axs[1].set_title('Model vs. Inference Duration', fontsize=6, pad=3)
    axs[1].set_ylabel(ylabel='Time (ms.)', fontsize=5)
    axs[1].tick_params(axis='both', pad=2, length=2, direction='inout', labelsize=5)

    plot = sns.barplot(x=model_names, y=accuracy_scores, ax=axs[2], palette='viridis')
    axs[2].bar_label(plot.containers[0], size=5)
    axs[2].set_title('Model vs. Prediction Accuracy', fontsize=6, pad=3)
    axs[2].set_ylabel(ylabel='Accuracy Score (%)', fontsize=5)
    axs[2].tick_params(axis='both', pad=2, length=2, direction='inout', labelsize=5)
    
    # Adjust subplot spacing
    plt.tight_layout()

    # Save the plot as a file
    plt.savefig(f'{directory}/MNIST_Model_Performances.png')

    # Show the plot
    plt.show()

    # Print the directory where the final comparison charts are saved
    print(f'\nFinal performance evaluations saved in ~{directory} directory.')


def train_and_compare_models(model_choices, X_train, y_train, X_test, y_test):
    # Initialize empty lists to store results
    training_durations = []
    inference_durations = []
    accuracy_scores = []
    model_names = []
    predictions = []

    # Scale the features of the training and test sets 
    X_train_scaled, X_test_scaled = scale_mnist_features(X_train, X_test)

    # Loop over the available model choices
    for model_choice in model_choices:
        if model_choice == 'KNC':
            # Instantiate the KNeighborsClassifier model
            model = KNeighborsClassifier()
        elif model_choice == 'LR':
            # Instantiate the LogisticRegression model
            model = LogisticRegression(multi_class='multinomial')
        elif model_choice == 'RFC':
            # Instantiate the RandomForestClassifier model
            model = RandomForestClassifier()
        elif model_choice == 'SVC':
            # Instantiate the SVC model
            model = SVC()
        # Get the name of the model
        model_name = model.__class__.__name__
        # Print information about training the model
        print(f'\nTraining the {model_name} model...')
        # Train the model and calculate training time
        t_train = train_model_and_calc_training_time(model, X_train_scaled, y_train)
        # Print information about the training duration (ms.)
        print(f'{model_name} model was trained in {t_train} Milliseconds.')
        # Save the trained model
        save_model(model)
        # Print information about predicting with the trained model
        print(f'\n{model_name} model is now predicting targets...')
        # Predict using the trained model and calculate inference time
        preds, t_infer = predict_targets_and_calc_prediction_time(model, X_test_scaled)
        # Print information about the inference duration (ms.)
        print(f'{model_name} model made predictions based on the test set in {t_infer} Milliseconds.')
        # Calculate the accuracy score of the model
        acc_score = np.round(accuracy_score(y_test, preds), 4)
        # Display and save the metrics of the model
        display_and_save_metrics(model, y_test, preds)

        # Append the metrics of the current model to their corresponding lists
        training_durations.append(t_train)
        inference_durations.append(t_infer)
        accuracy_scores.append(acc_score)
        model_names.append(model_name)
        predictions.append(preds)

    # Transpose the preds list, create a pandas DF with the transposed predictions and model names as cols
    transposed_predictions = np.transpose(predictions).tolist()
    predictions_df = pd.DataFrame(transposed_predictions, index=range(len(predictions[0])), columns=model_names)

    # Create a pandas dataframe with the original test set data
    test_df = pd.DataFrame(X_test)

    # Create a pandas dataframe of model names and their performance results
    model_performances = pd.DataFrame(columns=model_names,
                                      data=[training_durations,
                                            inference_durations,
                                            accuracy_scores])

    # Compare, display and save the final evaluation metrics of all the models
    compare_display_save_final_evaluations(model_performances)

    # Return test, predictions, and model performances dataframes.
    return test_df, y_test, predictions_df


def plot_digit(test_df, test_labels, predictions_df):
    # Get the length of the test dataframe
    n = len(test_df)
    # Generate a random index to pick a row from the test dataframe
    index = random.randint(0, n - 1)
    # Get the row corresponding to the random index
    row = test_df.iloc[index]
    # Convert the row to a numpy array and reshape it to 28x28 dimensions
    digit = np.array(row).reshape(28, 28)
    # Create a heatmap using seaborn to visualize the digit
    plot = sns.heatmap(digit, cmap='binary', cbar=False)
    # Set the xticks and yticks for the heatmap
    plot.set_xticks(range(0, 29, 2))
    plot.set_yticks(range(0, 29, 2))
    # Set the labels for the xticks and yticks
    plot.set_xticklabels(range(0, 29, 2), size=6)
    plot.set_yticklabels(range(0, 29, 2), size=6)
    # Remove the top, right, left and bottom spines from the plot
    sns.despine(top=False, right=False, left=False, bottom=False)
    # Set the title for the plot
    plt.title(label=f'Target (human-labeled): {test_labels[index]}', fontsize=10)
    # Create a mapping of model names to their abbreviations
    modelname_map = {'KNeighborsClassifier': 'KNC:', 'LogisticRegression': 'LR:',
                     'RandomForestClassifier': 'RFC:', 'SVC': 'SVC:'}
    # Rename the columns in the predictions dataframe using the model_name_map
    predictions_df = predictions_df.rename(columns=modelname_map)
    # Get the prediction label for the randomly selected digit
    title = predictions_df.iloc[index].to_string()
    # Create a legend for the plot with the prediction label as the title and label
    plt.legend(title=title, labels=[title], title_fontsize=7, loc=2, frameon=False)


def plot_random_digit_with_predictions(test_df, test_labels, predictions_df):
    # Define a figure with a size of 4 by 4 inches
    plt.figure(figsize=(4, 4))
    # Call the plot_digit function to plot a random digit with its prediction
    plot_digit(test_df, test_labels, predictions_df)
    # Print instructions for refreshing and closing the plot
    print('\nPress Space Tab to refresh the plot, Esc. key to close the figure window & terminate the program.')

    # Define a function that listens to key presses
    def on_key_press(event):
        # If the key pressed is the space bar, clear the current plot and redraw the digit with its prediction
        if event.key == ' ':
            plt.clf()
            plot_digit(test_df, test_labels, predictions_df)
            plt.draw()
        # If the key pressed is the escape key, close the plot window and exit the program
        elif event.key == 'escape':
            plt.close()
            sys.exit()

    # Connect the key press event to the on_key_press function
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)
    # Display the plot
    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = unfold_mnist_data()
    plot_class_counts(y_train, y_test)
    model_choices = get_model_choice()
    test_df, test_labels, preds_df = train_and_compare_models(model_choices, X_train, y_train, X_test, y_test)
    plot_random_digit_with_predictions(test_df, test_labels, preds_df)
