import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    'audi', 'ford', 'audi', 'toyota', 'ford', 'bmw',
    'toyota', 'ford', 'audi'])

encoder = sp.LabelEncoder()
lable_result = encoder.fit_transform(raw_samples)
print(lable_result)

source = encoder.inverse_transform(lable_result)
print("source :",source)