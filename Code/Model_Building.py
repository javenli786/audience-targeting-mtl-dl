import tensorflow as tf
def create_model(params, input_shape):
    """
    Create a multi-task learning model
    Args:
        params (dict): Hyperparameters for the model.
        input_shape (tuple): Shape of the input data.
    """
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(3):
        x = tf.keras.layers.Dense(params["units"], activation=params["activation"])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(params["dropout_rate"])(x)
    
    reg_output = tf.keras.layers.Dense(1, name='regression_output')(x)
    class_output = tf.keras.layers.Dense(1, activation='sigmoid', name='classification_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[reg_output, class_output])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss={
            'regression_output': 'mse',
            'classification_output': 'binary_crossentropy'
        },
        metrics={
            'regression_output': 'mae',
            'classification_output': tf.keras.metrics.Recall(name='recall')
        }
    )
    
    return model



def train_model(model, X_train, Y_train, X_val, Y_val, sample_weights):
    """
    Train the multi-task learning model.
    Args:
        model (tf.keras.Model): The multi-task learning model.
        X_train: Training data.
        Y_train: Target data for training.
        X_val: Validation data.
        Y_val: Target data for validation.
        sample_weights: Sample weights for training.
    """
    history = model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_classification_output_recall', 
                patience=5, 
                restore_best_weights=True
            )
        ],
        verbose=0,
        sample_weight=sample_weights
    )
    
    return history