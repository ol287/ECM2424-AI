import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_excel("/content/CCD (1).xls")

# Assuming the last column is the target variable and the rest are features
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target variable

# Convert all columns to numeric, coercing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')
# Fill NaN values with the median of each column, only for numeric columns
X.fillna(X.median(numeric_only=True), inplace=True)

# Ensure all columns in X are numeric
if not all(X.dtypes.apply(lambda dtype: np.issubdtype(dtype, np.number))):
    raise ValueError("Not all columns in the dataframe are numeric after conversion.")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.astype(np.float32)  # Convert to float32

# Convert any integer types in 'y' to strings for uniformity
if not all(isinstance(item, str) for item in y):
    y = y.astype(str)

# Encode the categorical target variable y
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert y_encoded to float32 for TensorFlow, as TensorFlow expects float inputs for targets
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

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
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')

# Model training
history = model.fit(
    X_train, y_train, epochs=100, validation_split=0.2,
    callbacks=[early_stopping], batch_size=32
)

# Plotting the accuracy and loss graphs
# Summarize history for accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # first plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Summarize history for loss
plt.subplot(1, 2, 2)  # second plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')
