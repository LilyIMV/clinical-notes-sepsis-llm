import tensorflow as tf
import tensorflow.keras.backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss

def build_lstm_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    lstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(x)

    attn_scores = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
    attn_flat = tf.keras.layers.Flatten()(attn_scores)
    attn_weights = tf.keras.layers.Activation('softmax', name="attn_weights")(attn_flat)
    attn_repeat = tf.keras.layers.RepeatVector(128)(attn_weights)
    attn_permute = tf.keras.layers.Permute([2, 1])(attn_repeat)
    attended = tf.keras.layers.multiply([lstm_out, attn_permute])
    attention_output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)

    max_pooled = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(lstm_out)
    combined = tf.keras.layers.concatenate([attention_output, max_pooled])

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model