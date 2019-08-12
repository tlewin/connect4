import math
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
        self.model.summary()

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
            32,
            (4, 4),
            input_shape=(c4.BOARD_HEIGHT, c4.BOARD_WIDTH, 2),
            padding='same',
            activation="relu",
        ))
        model.add(tf.compat.v1.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Conv2D(
            32,
            (4, 4),
            padding='same',
            activation="relu",
        ))
        model.add(tf.compat.v1.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Conv2D(
            32,
            (4, 4),
            padding='same',
            activation="relu",
        ))
        model.add(tf.compat.v1.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Flatten(input_shape=(c4.BOARD_HEIGHT, c4.BOARD_WIDTH, 2)))
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.compat.v1.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.compat.v1.keras.layers.LeakyReLU())
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.compat.v1.keras.layers.LeakyReLU())
        # model.add(tf.keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.Dense(16, activation="relu"))
        # model.add(tf.keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.Dense(
        #     c4.BOARD_WIDTH,
        #     # activation="relu",
        #     name="q_values",
        # ))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            # loss="mse",
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
            metrics=['accuracy', 'binary_crossentropy'],
        )
        if model_file:
            model.load_weights(model_file)
        return model


def board2tensor(board):
    return board.swapaxes(0, 2).swapaxes(0, 1)


def build_player2_fn(model, print_predictions=False):
    def player2_fn(board, player):
        plays = c4.available_plays(board)
        if player == 2:
            # import ipdb; ipdb.set_trace()
            board = c4.swap_players(board)
        p_values = model.model.predict(
            np.array([
                board2tensor(c4.play(board, 1, play))
                for play in plays
            ])
        )
        if print_predictions:
            print zip(plays, p_values.reshape(-1))
        play_index = np.argmax(p_values)
        play = plays[play_index]
        return play
    return player2_fn


def train(
    episodes=10000,
    gamma=0.30,
    epsilon=0.8,
    decay_epsilon=1.0,
    epsilon_decay=0.0,
    model_file=None,
    output_file=None):

    model = Model(model_file=model_file)

    def run_episode(epsilon):
        board = c4.new_board()
        play_history = []
        player = np.random.choice([1, 2])
        player_fn = build_player2_fn(model)

        while True:
            plays = c4.available_plays(board)
            if not plays:
                return (0, play_history,)

            if np.random.rand() < epsilon:
                play = np.random.choice(plays)
            else:
                play = player_fn(board, player)

            board = c4.play(board, player, play)

            input_state = None
            if player == 1:
                input_state = board2tensor(board)
            else:
                input_state = board2tensor(c4.swap_players(board))

            play_history.append((player, play, input_state))

            winner = c4.has_winner(board)
            if winner:
                return (winner, play_history,)

            player = 2 if player == 1 else 1


    train_inputs = []
    train_outputs = []
    for episode in xrange(episodes):
        if episode % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        # if (episode + 1) % 500 == 0:
        #     print ""
        #     c4.play_match(
        #         player1_fn=build_player2_fn(model, print_predictions=True), # c4.manual_player,
        #         player2_fn=build_player2_fn(model, print_predictions=True),
        #     )

        (winner, history) = run_episode(
            epsilon / math.sqrt(epsilon_decay + 1.0)
        )
        player1_outcome = 0
        player2_outcome = 0
        if winner == 1:
            player1_outcome = 1
            player2_outcome = 0
        elif winner == 2:
            player1_outcome = 0
            player2_outcome = 1
        inputs = []
        outputs = []
        for h in history:
            (player, play, state) = h
            # if player == 2:
            #     continue
            output = player1_outcome if player == 1 else player2_outcome
            inputs.append(state)
            outputs.append(output)

        train_inputs.extend(inputs)
        train_outputs.extend(outputs)
        if (episode + 1) % 100 == 0:
            model.model.fit(
                np.array(train_inputs),
                np.array(train_outputs),
                batch_size=32,
                shuffle=True,
            )
            epsilon_decay = 0.0
            train_inputs = []
            train_outputs = []
            if output_file:
                model.model.save_weights(output_file)
        epsilon_decay += 1.0

    c4.play_match(player1_fn=c4.manual_player, player2_fn=build_player2_fn(model))


def print_history(history):
    (player, play, state) = history
    print "Player: {} - column {}".format(player, play)
    print c4.print_board(state.swapaxes(0, 1).swapaxes(0, 2))


if __name__ == "__main__":
    model = Model(model_file="test3.h5")
    c4.play_match(player1_fn=c4.manual_player, player2_fn=build_player2_fn(model, print_predictions=True))
    # train(
    #     episodes=1,
    #     epsilon=0.9,
    #     gamma=1.0,
    #     decay_epsilon=1000,
    #     # model_file="test3.h5",
    #     output_file="test3.h5",
    # )
