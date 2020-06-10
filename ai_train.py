from Game.game import Game
import random
import numpy as np 
import tensorflow as tf
import copy
import time
import shaper
from os import system, name 
from time import sleep 
import sys
  

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def clear(): 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 
        

#TODO checking if previous moves were not the same  DONE
#TODO fitness from both games                       DONE
#TODO genetic reinforcment learning                 DONE
#TODO make it work                              not done


def print_board(positions):
    odd = True
    count = 1
    for element in positions:
 
        if odd:
            print("{}  {}  ".format(0, element), end = "")
            count += 1
        else:
            print("{}  {}  ".format(element, 0), end = "")
            count += 1
 
        if count > 4:
            count = 1
            odd = not odd
            print("\n")


def check_if_move_is_continous(previous_moves, move):
    same_moves = 0
    #print(previous_moves, move)
    for prev in previous_moves:
        if move[0] in prev and move[1] in prev:
            same_moves += 1
    if same_moves == 6:      
        return True
    else:
        return False

def create_numerical_board(prev_board, new_board):
    numerical_board = [0 for _ in range(64)]
    for piece in prev_board.pieces:
        
        if piece.position != None:
            
            if prev_board.player_turn == 2 and piece.player == 1:
                numerical_board[piece.position - 1] = 1
            elif prev_board.player_turn == 2 and piece.player == 2:
                numerical_board[piece.position - 1] = 2
            elif prev_board.player_turn == 1 and piece.player == 1:
                numerical_board[piece.position - 1] = 2
            elif prev_board.player_turn == 1 and piece.player == 2:
                numerical_board[piece.position - 1] = 1
                
    for piece in new_board.pieces:
        
        if piece.position != None:
            
            if new_board.player_turn == 2 and piece.player == 1:
                numerical_board[piece.position - 1 + 32] = 1
            elif new_board.player_turn == 2 and piece.player == 2:
                numerical_board[piece.position - 1 + 32] = 2
            elif new_board.player_turn == 1 and piece.player == 1:
                numerical_board[piece.position - 1 + 32] = 2
            elif new_board.player_turn == 1 and piece.player == 2:
                numerical_board[piece.position - 1 + 32] = 1
                
    return np.array([np.array(numerical_board)])

def create_numerical_board_to_print(new_board):
    numerical_board = [0 for _ in range(32)]
    for piece in new_board.pieces:
        
        if piece.position != None:
            numerical_board[piece.position - 1] = piece.player
            
    return np.array(numerical_board)

class AiModule(object):
    def __init__(self, N_PLAYERS = 20, N_KIDS = 5, N_OF_ITERATIONS = 30):
        self.player_type = ['white', 'black']
        self.N_PLAYERS = N_PLAYERS 
        self.N_OF_ITERATIONS = N_OF_ITERATIONS
        self.N_KIDS = N_KIDS
        players = [np.array(self.get_model().get_weights()) for _ in range(self.N_PLAYERS)]
        for player_index in range(len(players)):
            players[player_index] = [copy.deepcopy(players[player_index]), shaper.get_biases(players[player_index]), 0]
        self.players = players

    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(64))
        model.add(tf.keras.layers.Dense(128, activation = 'relu'))
        model.add(tf.keras.layers.Dense(128, activation = 'relu'))
        model.add(tf.keras.layers.Dense(64, activation = 'relu'))
        model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'mse', optimizer = 'adam')
        return model
    
    def play(self, players, game_index = None, show_game = False):
        game = Game()
        start_time = time.time()
        previous_moves = [[[0,0] for _ in range(6)] for _ in range(2)]
        while game.is_over() == False:
            possible_moves = copy.deepcopy(game.get_possible_moves())
            best_one = possible_moves[np.argmax(np.array([players[game.whose_turn() - 1].predict(create_numerical_board(game.board, game.board.create_new_board_from_move(move)))[0][0] for move in possible_moves]))]
            if show_game:
                clear()
                print_board(create_numerical_board_to_print(game.board))
                print('number of game:', game_index)
                sleep(0.1)
            curr_player = game.whose_turn()
            game.move(best_one)
            if check_if_move_is_continous(previous_moves[curr_player - 1], best_one) == True:
                return game.whose_turn(), 0.5, time.time() - start_time
            previous_moves[curr_player - 1] = previous_moves[curr_player - 1][1:] + [best_one]
        if show_game:
            clear()
            print_board(create_numerical_board_to_print(game.board))
            print('number of game:', game_index)
        return game.get_winner(), 1.0, time.time() - start_time
    
    
    
    def genetic_learning(self, show_games = False):
        total_n_of_games = self.N_PLAYERS * (self.N_KIDS + 1) * (self.N_PLAYERS * (self.N_KIDS + 1) - 1)

        for iteration in range(self.N_OF_ITERATIONS):
            current_players = []
            for player in self.players:
                current_players.append(copy.deepcopy(player))
                weights = player[0]
                biases = player[1]
                for _ in range(self.N_KIDS):
                    new_weights, new_biases = shaper.evolve(weights, biases)
                    current_players.append([new_weights, new_biases, 0])
            
            current_game = 1
            for player_index in range(len(current_players)):
                for enemy_index in range(len(current_players)):
                    if player_index != enemy_index:
                        player_white = self.get_model()
                        player_white.set_weights(np.array(current_players[player_index][0]))
                        player_black = self.get_model()
                        player_black.set_weights(np.array(current_players[enemy_index][0]))
                        winner, fitness, _ = self.play([player_white, player_black])
                        if winner == 1:
                            current_players[player_index][2] += fitness
                        elif winner == 2:
                            current_players[enemy_index][2] += fitness
                        del player_white, player_black
                        current_game += 1
                        printProgressBar(current_game,total_n_of_games, prefix = 'Progress:', suffix = 'Complete', length = 50)
                        
            players_sorted = [player for player in sorted(current_players, key = lambda x: x[2], reverse=True)]      
            self.players = copy.deepcopy(players_sorted[:self.N_PLAYERS])
            print('iteration: ', iteration, 'winning fitness: ',self.players[0][2])
            for player_index in self.players:
                self.players[player_index][2] = 0  
            
         





        # games = [[[0,0],[0,0]] for _ in range(int(self.N_PLAYERS/2))]
        # for iteration in range(self.N_OF_ITERATIONS):
        #     print('iteration: ', iteration)
        #     print('0%')
        #     for game_index in range(int(self.N_PLAYERS/2)):
        #         if np.random.uniform(0,1) < 0.1 and show_games:
        #             winner, fitness, game_time = self.play([self.players[2*game_index], self.players[2*game_index + 1]], game_index = game_index, show_game=True)
        #         else:
        #             winner, fitness, game_time = self.play([self.players[2*game_index], self.players[2*game_index + 1]], game_index = game_index)
        #         games[game_index][0][0] = winner
        #         games[game_index][0][1] = fitness
        #         if winner != None:
        #             print('winner:', self.player_type[winner - 1], '| fitness:', fitness, '| time:', game_time)
        #         else:
        #             print('winner:', 'None', '| fitness:', fitness, '| time:', game_time)
        #         sys.stdout.flush()  

        #     print('50%')      

        #     for game_index in range(int(self.N_PLAYERS/2)):
        #         if np.random.uniform(0,1) < 0.1 and show_games:
        #             winner, fitness, game_time = self.play([self.players[2*game_index + 1], self.players[2*game_index ]], game_index = game_index, show_game=True)
        #         else:
        #             winner, fitness, game_time = self.play([self.players[2*game_index + 1], self.players[2*game_index ]], game_index = game_index)
        #         games[game_index][1][0] = winner
        #         games[game_index][1][1] = fitness
        #         if winner != None:
        #             print('winner:', self.player_type[winner - 1], '| fitness:', fitness, '| time:', game_time)
        #         else:
        #             print('winner:', 'None', '| fitness:', fitness, '| time:', game_time)
                
        #     parents = []
        #     for game_index in range(len(games)):
        #         if games[game_index][0][0] == games[game_index][1][0] == 1:
        #             if games[game_index][0][1] > games[game_index][1][1]:
        #                 parents.append(self.players[2*game_index])
        #                 continue
        #             else:
        #                 parents.append(self.players[2*game_index + 1])
        #                 continue
        #         elif games[game_index][0][0] == games[game_index][1][0] == 2:
        #             if games[game_index][0][1] > games[game_index][1][1]:
        #                 parents.append(self.players[2*game_index + 1])
        #                 continue
        #             else:
        #                 parents.append(self.players[2*game_index])
        #                 continue
        #         elif games[game_index][0][0] == 1 and  games[game_index][1][0] == 2 or games[game_index][0][0] == 1 and  games[game_index][1][0] == None:
        #             parents.append(self.players[2*game_index])
        #             continue
        #         elif games[game_index][0][0] == 2 and  games[game_index][1][0] == 1 or games[game_index][0][0] == 2 and  games[game_index][1][0] == None:
        #             parents.append(self.players[2*game_index + 1])
        #             continue
        #         elif games[game_index][0][0] == None and  games[game_index][1][0] == None:
        #             parents.append(self.players[2*game_index + random.randint(0,1)])
        #             continue
                
        #     print('100%')
        #     new_players = []
        #     print('crossover and mutation')
        #     for _ in range(self.N_PLAYERS - len(parents)):
        #         parent1 = parents[random.randint(0,len(parents) - 1)]
        #         parent2 = parents[random.randint(0,len(parents) - 1)]
        #         weights = parent1.get_weights()
        #         for i in range(len(weights)):
        #             if np.random.uniform(0, 1) > 0.5:
        #                 weights[i] = parent2.get_weights()[i]
        #         new_baby = self.get_model()
        #         weights = y4ndhi(weights)
        #         new_baby.set_weights(weights)
        #         new_players.append(tf.keras.models.clone_model(new_baby))
        #         del new_baby
        #     for parent in parents:
        #         new_players.append(tf.keras.models.clone_model(parent))
        #     del self.players
        #     random.shuffle(new_players)
        #     self.players = new_players
        #     self.players[0].save_weights('./training/weights')





          
training = AiModule(N_PLAYERS=10,N_KIDS=2 ,N_OF_ITERATIONS=50)
training.genetic_learning()


    
    
    


            
        
        
