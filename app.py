from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the model
model = load_model("Model/keras_model.h5")

# Compile the model manually
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Replace with the optimizer of your choice
    loss="categorical_crossentropy",     # Replace with the appropriate loss function
    metrics=["accuracy"]                 # Add metrics as needed
)

# Continue training the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
