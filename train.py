import math
import random
import sys

import numpy as np
import tensorflow as tf

import connect4 as c4


class Model:

    def __init__(self,
                 learning_rate=0.0001,
                 model_file=None):
        self.learning_rate = learning_rate
        self.model = self._build_model(model_file=model_file)

    def _build_model(self, model_file=None):
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Dense(
        #     c4.BOARD_HEIGHT * c4.BOARD_WIDTH * 2,
        #     input_shape=(c4.BOARD_HEIGHT * c4.BOARD_WIDTH * 2,),
        #     name="input",
        # ))
        # # NOTE: There are 69 possible ways, for each player, to connect in 4
        # model.add(tf.keras.layers.Dense(
        #     150,
        #     activation="relu",
        #     # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1),
        # ))
        # model.add(tf.keras.layers.Dense(
        #     150,
        #     activation="relu",
        #     # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1),
        # ))
        model.add(tf.keras.layers.Conv2D(
            128,
            (4, 4),
            input_shape=(c4.BOARD_WIDTH, c4.BOARD_HEIGHT, 2),
            padding='same',
            activation="relu",
        ))
        model.add(tf.keras.layers.Conv2D(
            128,
            (4, 4),
            padding='same',
            activation="relu",
        ))
        model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(32, activation="relu"))
        model.add(tf.keras.layers.Dense(32, activation="relu"))
        # model.add(tf.keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.Dense(16, activation="relu"))
        # model.add(tf.keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.Dense(
        #     c4.BOARD_WIDTH,
        #     # activation="relu",
        #     name="q_values",
        # ))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
        )
        if model_file:
            model.load_weights(model_file)
        return model


def board2tensor(board):
    return board.transpose()

def build_player2_fn(model, print_predictions=False):
    def player2_fn(board, player):
        plays = c4.available_plays(board)
        if player == 1:
            q_values = model.model.predict(
                np.array([board2tensor(board)])
            )[0]
        else:
            q_values = model.model.predict(
                np.array([board2tensor(c4.swap_players(board))])
            )[0]
        if print_predictions:
            print q_values
        play = np.argmax(q_values)
        # NOTE: very unlike to happend
        if play not in plays:
            ranked_plays = sorted(enumerate(q_values), key=lambda t: t[-1], reverse=True)
            # NOTE: The first one it's already known that is not available
            # anymore.
            for index, _ in ranked_plays[1:]:
                if index in plays:
                    play = index
                    break
        return play
    return player2_fn


def train(
    episodes=10000,
    gamma=0.30,
    epsilon=0.8,
    decay_epsilon=1.0,
    model_file=None,
    output_file=None):

    model = Model(model_file=model_file)

    def run_episode(epsilon):
        board = c4.new_board()
        play_history = []
        player = random.choice([1, 2])

        while not c4.has_winner(board):
            plays = c4.available_plays(board)
            if not plays:
                return (0, play_history,)

            player = 2 if player == 1 else 1
            input_state = None
            if player == 1:
                input_state = board2tensor(board)
            else:
                input_state = board2tensor(c4.swap_players(board))

            q_values = model.model.predict(np.array([input_state]))[0]
            invalid = False
            if random.random() < epsilon:
                play = random.choice(plays)
            else:
                play = np.argmax(q_values)
                # NOTE: very unlike to happen
                if play not in plays:
                    invalid = True
                    # ranked_plays = sorted(enumerate(q_values), key=lambda t: t[-1], reverse=True)
                    # # NOTE: The first one it's already known that is not available
                    # # anymore.
                    # for index, _ in ranked_plays[1:]:
                    #     if index in plays:
                    #         play = index
                    #         break

            play_history.append((player, play, input_state, q_values))
            if invalid:
                return (2 if player == 1 else 1, play_history,)
            board = c4.play(board, player, play)

        return (player, play_history,)

    train_inputs = []
    train_outputs = []
    for episode in xrange(episodes):
        if episode % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        if (episode + 1) % 1000 == 0:
            print ""
            c4.play_match(
                player1_fn=build_player2_fn(model, print_predictions=True), # c4.manual_player,
                player2_fn=build_player2_fn(model, print_predictions=True),
            )

        (winner, history) = run_episode(
            epsilon / math.sqrt((episode / decay_epsilon) + 1.0)
        )
        reward = 0
        if winner == 1:
            reward = 1
        elif winner == 2:
            reward = -1

        (player, play, state, q_values) = history[-1]
        k = -1
        if player == 1:
            inputs = [state]
            q_values[play] = reward
            outputs = [q_values]
        else:
            (player, play, state, q_values) = history[-2]
            inputs = [state]
            q_values[play] = reward
            outputs = [q_values]
            k = -2
        last_qvalues = q_values
        for h in reversed(history[:k]):
            (player, play, state, q_values) = h
            if player == 1:
                reward = gamma * np.max(last_qvalues)
            else:
                continue
                reward = -reward
            q_values[play] = reward
            last_qvalues = q_values
            inputs.insert(0, state)
            outputs.insert(0, q_values)
        if (episode + 1) % 10 == 0:
            model.model.fit(np.array(train_inputs), np.array(train_outputs), shuffle=True)
            train_inputs = []
            train_outputs = []
        else:
            train_inputs.extend(inputs)
            train_outputs.extend(outputs)

    if output_file:
        model.model.save_weights(output_file)
    c4.play_match(player1_fn=c4.manual_player, player2_fn=build_player2_fn(model))


def print_history(history):
    (player, play, state, q_values) = history
    print "Player: {} - column {}".format(player, play)
    print q_values
    print c4.print_board(state.reshape(2, c4.BOARD_HEIGHT, c4.BOARD_WIDTH))


if __name__ == "__main__":
    # model = Model(model_file="test3.h5")
    # c4.play_match(player1_fn=c4.manual_player, player2_fn=build_player2_fn(model))
    train(
        episodes= 20000,
        epsilon=0.8,
        gamma=1.0,
        decay_epsilon=500,
        # model_file="test3.h5",
        output_file="test3.h5",
    )
