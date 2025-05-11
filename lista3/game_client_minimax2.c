#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h> 

#include "./board.h" 

#define MAX_SCORE 1000000

int playerNumber;
int depthOfSearch;

// Evaluates the current board state from perspective of given player
int evaluateBoard(int player) {
    int opponent = 3 - player;
  
    // Check for immeidiate win/loss
    if (winCheck(player)) {
      return MAX_SCORE;
    }
    if (winCheck(opponent)) {
      return -MAX_SCORE;
    }
    if (loseCheck(player)) {
      return -MAX_SCORE;
    }
    if (loseCheck(opponent)) {
      return MAX_SCORE;
    }
  
    // Return heurisitic evaluation score
    return heuristicScore(player);
}
  
// Heuristic evaluation function based on the number of 4s and 3s
int heuristicScore(int player) {
    int score = 0;
    int opponent = 3 - player;

    // High score for achieving four-in-a-row, penalty for opponent achieving the same
    score += countFours(player) * 10000;
    score -= countFours(opponent) * 15000;

    // Negative score for three-in-a-row (own), positive for opponent's three-in-a-row
    score -= countThrees(player) * 1000;
    score += countThrees(opponent) * 500;

    return score;
}

// Counts the number of almost complete lines (three in a row with one empty cell)
int countFours(int player) {
    int count = 0;
    for (int i = 0; i < 28; i++) {
        int occupied = 0;
        int empty = 0;
        int blocked = 0;
        for (int j = 0; j < 4; j++) {
        int x = win[i][j][0];
        int y = win[i][j][1];

        if (board[x][y] == player) {
            occupied++;
        }
        else if (board[x][y] == 0) {
            empty++;
        }
        else {
            blocked = 1;
            break;
        }
        }

        if (!blocked && occupied == 3 && empty == 1) {
        count++;
        }
    }
    return count;
}

// Counts the number of losing lines (three in a row, all occupied by player)
int countThrees(int player) {
    int count = 0;
    for (int i = 0; i < 48; i++) {
        int occupied = 0;
        for (int j = 0; j < 3; j++) {
        int x = lose[i][j][0];
        int y = lose[i][j][1];

        if (board[x][y] == player) {
            occupied++;
        }
        }
        if (occupied == 3) {
            count++;
        }
    }
    return count;
}

// Checks whether a given move is valid (inside board boundaries and on an empty cell)
bool isMoveValid(int move) {
    int row = (move / 10) - 1;
    int col = (move % 10) - 1;

    if (row < 0 || row > 4 || col < 0 || col > 4) {
        return false;
    }
    return board[row][col] == 0;
}

// Counts the number of empty cells on the board
int countEmptyCells() {
    int count = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (board[i][j] == 0) {
                count++;
            }
        }
    }
    return count;
}

// Minimax algorithm with alpha-beta pruning to calculate optimal move
int minimax(int depth, bool maximizing, int player, int alpha, int beta) {
    int opponent = 3 - player;
    if (depth == 0 || winCheck(player) || loseCheck(player) || winCheck(opponent) || loseCheck(opponent)) {
        return evaluateBoard(playerNumber);
    }

    int bestVal = maximizing ? -MAX_SCORE : MAX_SCORE;

    for (int row = 1; row <= 5; row++) {
        for (int col = 1; col <= 5; col++) {
            int move = row * 10 + col;
            if (!isMoveValid(move)) continue;

            // Apply the move and call minimax recursively
            board[row - 1][col - 1] = player;
            int value = minimax(depth - 1, !maximizing, opponent, alpha, beta);
            board[row - 1][col - 1] = 0;

            if (maximizing) {
                if (value > bestVal) {
                    bestVal = value;
                }
                if (bestVal > alpha) {
                    alpha = bestVal;
                }
            } else {
                if (value < bestVal) {
                    bestVal = value;
                }
                if (bestVal < beta) {
                    beta = bestVal;
                }
            }

            if (beta <= alpha) {
                break;
            };
        }
    }

    return bestVal;
}

// Determines the best move for the current player using the minimax algorithm
int getBestMove(int player, int depth) {
    // Prefer middle and corners on the beginning of the game
    int middle = 33;
    int corners[] = {11, 15, 51, 55};
    if (countEmptyCells() >= 24) {
        if (isMoveValid(middle)) {
            return middle;
        }
        for (int i = 0; i < 4; i++) {
            if (isMoveValid(corners[i])) {
                return corners[i];
            }
        }
    }
    int bestVal = -MAX_SCORE;
    int bestMoves[25];
    int count = 0;

    // Evaluate all possible moves and find the best one
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= 5; j++) {
            int move = i * 10 + j;
            if (!isMoveValid(move)) continue;

            // Simulate move and evaluate
            board[i - 1][j - 1] = player;
            int value = minimax(depth - 1, false, 3 - player, -MAX_SCORE, MAX_SCORE);
            board[i - 1][j - 1] = 0;

            // Update the best move
            if (value > bestVal) {
                bestVal = value;
                bestMoves[0] = move;
                count = 1;
            } else if (value == bestVal) {
                bestMoves[count++] = move;
            }
        }
    }

    // If there are multiple best moves, choose one randomly
    if (count > 0) {
        srand(time(NULL));
        return bestMoves[rand() % count];
    }

    return -1; // Return -1 if no valid moves are found
}


int main(int argc, char *argv[]) {
    if (argc != 6) {
        printf("Usage: %s <ip> <port> <player_num> <player_name> <depth>\n", argv[0]);
        return -1;
    }

    if (strlen(argv[4]) == 0 || strlen(argv[4]) > 9) {
        printf("Error: Player name must be 1–9 characters\n");
        return -1;
    }

    playerNumber = atoi(argv[3]);
    if (playerNumber != 1 && playerNumber != 2) {
        printf("Error: Player number must be 1 or 2\n");
        return -1;
    }

    depthOfSearch = atoi(argv[5]);
    if (depthOfSearch < 1 || depthOfSearch > 10) {
        printf("Error: Depth must be 1–10\n");
        return -1;
    }

    int server_socket;
    struct sockaddr_in server_addr;
    char server_message[16], player_message[16];
    
    bool end_game = false;
    int msg, move;

    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        printf("Unable to create socket\n");
        return -1;
    }
    printf("Socket created successfully\n");

    // Set port and IP the same as server-side
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(atoi(argv[2]));
    server_addr.sin_addr.s_addr = inet_addr(argv[1]);

    // Send connection request to server
    if (connect(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        printf("Unable to connect\n");
        return -1;
    }
    printf("Connected with server successfully\n");

    // Receive the server message
    memset(server_message, '\0', sizeof(server_message));
    // Receive server message
    if (recv(server_socket, server_message, sizeof(server_message), 0) < 0) {
        printf("Error while receiving server's message\n");
        return -1;
    }
    printf("Server message: %s\n", server_message);

    memset(server_message, '\0', sizeof(server_message));
    snprintf(player_message, sizeof(player_message), "%s %s", argv[3], argv[4]);
    // Send the message to server
    if ( send(server_socket, player_message, strlen(player_message), 0) < 0 ) {
        printf("Unable to send message\n");
        return -1;
    }

    setBoard();

    while (!end_game) {
        memset(server_message, '\0', sizeof(server_message));
        if (recv(server_socket, server_message, sizeof(server_message), 0) < 0) {
            printf("Error while receiving server's message\n");
            return -1;
        }
        printf("Server message: %s\n", server_message);
        sscanf(server_message, "%d", &msg);
        move = msg % 100;
        msg = msg / 100;

        if (move != 0) {
            setMove(move, 3 - playerNumber);
            printf("Opponent move: %d\n", move);
            printBoard();
        }

        if (msg == 0 || msg == 6) {
            move = getBestMove(playerNumber, depthOfSearch);
            printf("Chosen move: %d\n", move);
            setMove(move, playerNumber);
            printBoard();
            memset(player_message, '\0', sizeof(player_message));
            snprintf(player_message, sizeof(player_message), "%d", move);
            if (send(server_socket, player_message, strlen(player_message), 0) < 0) {
                printf("Unable to send message\n");
                return -1;
            }
            printf("Player message: %s\n", player_message);
        } else {
            end_game = true;
            switch (msg) {
                case 1: printf("You won.\n"); break;
                case 2: printf("You lost.\n"); break;
                case 3: printf("Draw.\n"); break;
                case 4: printf("You won. Opponent error.\n"); break;
                case 5: printf("You lost. Your error.\n"); break;
            }
        }
    }

    // Close socket
    close(server_socket);

    return 0;
}
