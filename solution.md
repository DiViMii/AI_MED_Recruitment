# Solution overview

## Task completion

With the help of the provided example I wrote code in 
python that:
1. Loads the data from the `task_data.csv` file. 
2. converts data in columns`CTR - Cardiothoracic Ratio`, `Inscribed circle radius`, `Heart perimeter` to numeric types.
3. Splits the data into training and test sets.
4. Performs standardization.
5. Trains one model.
6. Evaluates the solution using cross-validation
7. Evaluates the solution on the test dataset using accuracy, precision, recall, F1-score

## Training the model

I first used the provided example as a guide and trained a 
model — K-Nearest Neighbors classifier.
Then I attempted to optimise it resulting in the numbers of neighbours of `5`, 
weights as `uniform` and metric being `manhattan`.

## Evaluating the model

To evaluate the trained model i wrote code that outputs accuracy, precision, recall 
and F1-score as suggested in the task description.
Afterward I also wrote code that displays the ROC and Precision-Recall curves.