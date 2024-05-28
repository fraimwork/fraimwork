import tensorflow as tf
from keras import Input, LSTM, Dense
from tensorflow.keras.models import Model

# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


from tensorflow.keras.layers import AdditiveAttention

attention = AdditiveAttention()
context_vector, attention_weights = attention([decoder_outputs, encoder_outputs], return_attention_scores=True)
decoder_combined_context = tf.concat([context_vector, decoder_outputs], axis=-1)
