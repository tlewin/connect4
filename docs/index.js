require("babel-core/register");
require("babel-polyfill");

import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const BOARD_HEIGHT = 6;
const BOARD_WIDTH = 7;
const MODEL_URL = 'model.json';
let model;
var humanPlay;

function newBoard() {
    return [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]];
}

async function loadModel() {
    const model = await tf.loadLayersModel(MODEL_URL, {weightPathPrefix: '/connect4/'});
    return model;
}

function hasWinner(board) {
    for(var player = 0; player < 2; player++) {
        for(var h = 0; h < BOARD_HEIGHT; h++) {
            for(var w = 0; w < BOARD_WIDTH - 3; w++) {
                if( board[h][w][player] == 0 ) { continue }
                else if(
                    board[h][w][player] == board[h][w + 1][player]
                    && board[h][w][player] == board[h][w + 2][player]
                    && board[h][w][player] == board[h][w + 3][player]
                ) {
                    return player + 1;
                }
            }
        }

        for(var h = 0; h < BOARD_HEIGHT - 3; h++) {
            for(var w = 0; w < BOARD_WIDTH; w++) {
                if( board[h][w][player] == 0 ) { continue }
                else if(
                    board[h][w][player] == board[h + 1][w][player]
                    && board[h][w][player] == board[h + 2][w][player]
                    && board[h][w][player] == board[h + 3][w][player]
                ) {
                    return player + 1;
                }
            }
        }

        for(var h = 0; h < BOARD_HEIGHT - 3; h++) {
            for(var w = 0; w < BOARD_WIDTH - 3; w++) {
                if( board[h][w][player] == 0 ) { continue }
                else if(
                    board[h][w][player] == board[h + 1][w + 1][player]
                    && board[h][w][player] == board[h + 2][w + 2][player]
                    && board[h][w][player] == board[h + 3][w + 3][player]
                ) {
                    return player + 1;
                }
            }
        }

        for(var h = 0; h < BOARD_HEIGHT - 3; h++) {
            for(var w = BOARD_WIDTH - 1; w > 2; w--) {
                if( board[h][w][player] == 0 ) { continue }
                else if(
                    board[h][w][player] == board[h + 1][w - 1][player]
                    && board[h][w][player] == board[h + 2][w - 2][player]
                    && board[h][w][player] == board[h + 3][w - 3][player]
                ) {
                    return player + 1;
                }
            }
        }
    }
    return 0;
}

function deepClone(board) {
    return JSON.parse(JSON.stringify(board));
}

function availablePlays(board) {
    var plays = [];
    for(var i = 0; i < BOARD_WIDTH; i++) {
        if(
            board[BOARD_HEIGHT - 1][i][0] == 0
            && board[BOARD_HEIGHT - 1][i][1] == 0
        ) {
            plays.push(i);
        }
    }
    return plays;
}

function playMove(board, player, column) {
    return playMove_(deepClone(board), player, column);
}

function playMove_(board, player, column) {
    if(
        board[BOARD_HEIGHT - 1][column][0]
        || board[BOARD_HEIGHT - 1][column][1]
    ) { return null; }

    var row = BOARD_HEIGHT - 1;
    while( row >= 0 ) {
        if(board[row][column][0] || board[row][column][1]) { break }
        row--;
    }

    board[row + 1][column][player - 1] = 1;
    return board;
}

async function computerPlay(model, board) {
    const plays = availablePlays(board);
    const states = plays.map((play) => playMove(board, 1, play));
    const predictions = await model.predict(tf.tensor4d(states), { verbose: true });
    const buffer = predictions.bufferSync();

    const argmax = plays.reduce((acc, value) => {
        if( buffer.get(value) > buffer.get(acc) ) {
            return value;
        }
        return acc;
    }, 0);
    return plays[argmax];
}

function renderBoard(board) {
    for(var h = 0; h < BOARD_HEIGHT; h++) {
        for(var w = 0; w < BOARD_WIDTH; w++) {
            const elemId = "token-" + (h + 1) + "-" + (w + 1);
            const elem = document.getElementById(elemId);
            if( board[h][w][0] ) {
                elem.classList.add("player1");
            } else if( board[h][w][1] ) {
                elem.classList.add("player2");
            }
            else {
                elem.classList.remove("player1");
                elem.classList.remove("player2");
            }
        }
    }
}

async function start() {
    model = await loadModel();
    while( true ) {
        const board = newBoard();
        renderBoard(board);
        var player = Math.random() > 0.5 ? 1 : 2;
        var result;

        while(true) {
            result = hasWinner(board);

            if( result ) { break; }
            let play;
            const plays = availablePlays(board);
            if(plays.length === 0) {
                result = 0;
                break;
            }

            if( player === 1 ) {
                play = await computerPlay(model, board);
            } else {
                humanPlay = 0;
                var humanPromise = new Promise((resolve, reject) => {
                    const id = setInterval(() => {
                        if( humanPlay ) {
                            clearInterval(id);
                            resolve(humanPlay - 1);
                        }
                    }, 50);
                });
                play = await humanPromise;
            }
            playMove_(board, player, play);
            renderBoard(board);
            player = player == 1 ? 2 : 1;
        }
        let message;
        if( result == 0 ) { message = "No one won. Play again?"; }
        else if( result == 1 ) { message = "Alpha connect4 won. Play again?"; }
        else { message = "Congratulations! Play again?"; }
        var confirmPromise = new Promise((resolve, reject) => {
            setTimeout(() => {
                resolve( confirm(message) );
            })
        });
        var playAgain = await confirmPromise;
        if( !playAgain ) { break }
    }
}

document.getElementById("column1").addEventListener("click", () => {
    humanPlay = 1;
});
document.getElementById("column2").addEventListener("click", () => {
    humanPlay = 2;
});
document.getElementById("column3").addEventListener("click", () => {
    humanPlay = 3;
});
document.getElementById("column4").addEventListener("click", () => {
    humanPlay = 4;
});
document.getElementById("column5").addEventListener("click", () => {
    humanPlay = 5;
});
document.getElementById("column6").addEventListener("click", () => {
    humanPlay = 6;
});
document.getElementById("column7").addEventListener("click", () => {
    humanPlay = 7;
});

start();
