import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_excel("/content/CCD (1).xls")

# Convert all columns to numeric, coercing errors to NaN
X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
y = df.iloc[:, -1]

# Fill NaN values with the median of each column, only for numeric columns
X.fillna(X.median(numeric_only=True), inplace=True)

# Ensure all columns in X are numeric now
if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
    raise ValueError("Not all columns in the dataframe are numeric after conversion.")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model design
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')
