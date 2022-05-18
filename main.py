from sklearn.linear_model import LinearRegression as lr
import math
import matplotlib.pyplot as plt

# Predicts the RH based off FFMC DMC DC ISI and temp
file = open("forestfires.csv")

header = file.readline()

# Create two empty lists, this will be the input and output
X = []
y = []

# Read through the file and get the numeric values needed for input and output
for line in file:
  #RH
  RHDATA = line.split(",")[9]
  y.append(float(RHDATA))
  #FFMC DMC DC ISI and temp
  OTHERDATA = line.strip().split(",")[4:9]
  # Create an empty list and append that to input list
  temparr = []
  for i in range(len(OTHERDATA)):
    temparr.append(float(OTHERDATA[i]))
  X.append(temparr)

# Create empty  lists for training and testing
X_train = []
X_test = []
y_train = []
y_test = []

# split data into training and testing
for i in range(len(y)):
  if i < math.floor(len(y) * 0.8):
    X_train.append(X[i])
    y_train.append(y[i])
  else:
    X_test.append(X[i])
    y_test.append(y[i])

# train test predict
predictor = lr(n_jobs=-1)
predictor.fit(X=X_train, y=y_train)
outcome = []
coefficients = predictor.coef_

for tested in X_test:
  outcome.append(predictor.predict(X=[tested]))

# Get percentage error
percentError = []
for i in range(len(X_test)):
  if y_test[i] != 0:
    percentError.append(abs((y_test[i] - outcome[i]) / y_test[i]) * 100)

graphVal = [0,0,0,0,0,0,0,0,0,0,0]

for val in percentError:
  if val<10:
    graphVal[0] += 1
  elif 10<val<20:
    graphVal[1] += 1
  elif 20<val<30:
    graphVal[2] += 1
  elif 30<val<40:
    graphVal[3] += 1
  elif 40<val<50:
    graphVal[4] += 1
  elif 50<val<60:
    graphVal[5] += 1
  elif 60<val<70:
    graphVal[6] += 1
  elif 70<val<80:
    graphVal[7] += 1
  elif 80<val<90:
    graphVal[8] += 1
  else:
    graphVal[9] += 1

print(outcome)
print(coefficients)
print(percentError)
print(graphVal)


plt.xlabel("0% - 100%")
plt.ylabel("How many times error happened")
plt.title("Percent Error")
plt.plot(graphVal)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.show()