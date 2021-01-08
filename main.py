import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# Dataset and global parameters
df = pd.read_csv("data_PimaIndiansDiabetes.csv", delimiter=";")
df_filtered = df
test_proportion = 0.2


# We filter rows that contains O value for triceps and insulin because they are empty values (Manually)
df_filtered = df_filtered[df_filtered.insulin != 0]
df_filtered = df_filtered[df_filtered.triceps != 0]

# We filter the columns that contains too many O values (Manually)
df_filtered = df_filtered.drop(columns=['insulin'])
df_filtered = df_filtered.drop(columns=['triceps'])


# Get dimension before filters
nbRows_df = df.shape[0]
nbColumns_df = df.shape[1]

# Get dimension after filters
nbRows_df_filtered = df_filtered.shape[0]
nbColumns_df_filtered = df_filtered.shape[1]

# Display some characteristics of our analysis
print("After filters, number of rows  : ", nbRows_df_filtered, "/", nbRows_df, "(i.e. ",
      "%.2f" % round(nbRows_df_filtered/nbRows_df * 100, 2), "%)\nAfter filters, number of columns : ",
      nbColumns_df_filtered, "/", nbColumns_df, "(i.e. %.2f" % round(nbColumns_df_filtered/nbColumns_df * 100, 2),
      "%)\nPercentage of data we actually work on : ", "% .2f" % round(100*(nbRows_df_filtered*nbColumns_df_filtered)
                                                                       / (nbRows_df*nbColumns_df), 2), "%\n")

# We split inputs and output
nb_columns = df_filtered.shape[1]

X = df_filtered.iloc[:, 0:nb_columns-1].values
y = df_filtered.iloc[:, nb_columns-1].values

# standardising our data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion)

print("Training set size : ", y_train.size)
print("Test set size : ", y_test.size, "\n")


# Building the model, fitting data and make predictions
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix : \n", cm, "\n")

# Recall result that shows our model quality
recall = cm[0][0] / (cm[0][0] + cm[1][0])
print("Recall : ", cm[0][0], "/ (", cm[0][0], "+", cm[1][0], ") = ", "%.3f" % round(recall, 3))
