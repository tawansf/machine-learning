import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

load_dotenv()
URL_FILE = os.getenv('URL_FILE')+'wine_dataset.csv'

file = pd.read_csv(URL_FILE)

file['style'] = file['style'].replace('red', 0)
file['style'] = file['style'].replace('white', 1)

y = file['style']
x = file.drop('style', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = ExtraTreesClassifier()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)

sample_x = x_test[300:305]
sample_y = y_test[300:305]

prev = model.predict(sample_x)

print('Accuracy:', model.score(x_test, y_test))
print('Actual:', y_test[300:305])
print('Predicted:', prev)
