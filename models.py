from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

def build_basic_network(observation_size, action_size, learning_rate):
    """Builds a basic neural network architecture."""
    model = Sequential()
    model.add(Dense(24, input_dim=observation_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model
    