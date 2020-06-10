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
import json
import codecs  

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from lorem.text import TextLorem

lorem = TextLorem(srange=(1,2))
used_names = []
def create_name(species_name = '', original = False):
    if original:
        while True:
            name = lorem.sentence().replace('.', '')
            if name not in used_names:
                return species_name + name
    else:
        return species_name + lorem.sentence().replace('.', '')

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
#TODO make it work                              maybe?


def print_board(positions):
    """
    Prints the checkers board
    @params:
        positions   - Required  : current positions of pices on board (Int[])
    """
    odd = True
    count = 1
    board = '\r'
    for element in positions:
        
        if odd:
            board += "{}  {}  ".format(0, element)
        else:
            board += "{}  {}  ".format(element, 0)
        count += 1
 
        if count > 4:
            count = 1
            odd = not odd
            board += "\n"

    print(board, end='\r')


def check_if_move_is_continous(previous_moves, move):
    """
    Function that checks if the player is not repeating moves
    @params:
        previous_moves - Required  : list of previous moves (Int[][])
        move           - Required  : current move (Int[])  
    """
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
    """
    Function that creates input for Neurak Network,
    which is a list of all positions of pieces on thee board
    before and and after move
    @params:
        previous_moves - Required  : list of previous moves (Int[][])
        move           - Required  : current move (Int[])  
    """
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
    """
    Changes NN inputso that it can be printed
    @params:
        previous_moves - Required  : list of previous moves (Object)
    """
    numerical_board = [0 for _ in range(32)]
    for piece in new_board.pieces:
        
        if piece.position != None:
            numerical_board[piece.position - 1] = piece.player
            
    return np.array(numerical_board)

class AiModule(object):
    """
    Class responsible for training the Neural Network, 
    storing all players and 
    
    """
    def __init__(self, N_PLAYERS = 20, N_KIDS = 5, N_OF_ITERATIONS = 30):
        self.player_type = ['white', 'black']
        self.N_PLAYERS = N_PLAYERS 
        self.N_OF_ITERATIONS = N_OF_ITERATIONS
        self.N_KIDS = N_KIDS
        players = [np.array(self.get_model().get_weights()) for _ in range(self.N_PLAYERS)]
        for player_index in range(len(players)):
            players[player_index] = [copy.deepcopy(players[player_index]), shaper.get_biases(players[player_index]), 0, create_name(original=True)]
        self.players = players

    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(64))
        model.add(tf.keras.layers.Dense(256, activation = 'relu'))
        model.add(tf.keras.layers.Dense(128, activation = 'relu'))
        model.add(tf.keras.layers.Dense(64, activation = 'relu'))
        model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'mse', optimizer = 'adadelta')
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
                sleep(0.1)
            curr_player = game.whose_turn()
            game.move(best_one)
            if check_if_move_is_continous(previous_moves[curr_player - 1], best_one) == True:
                return game.whose_turn(), 0.5, time.time() - start_time
            previous_moves[curr_player - 1] = previous_moves[curr_player - 1][1:] + [best_one]
        if show_game:
            clear()
            print_board(create_numerical_board_to_print(game.board))
        return game.get_winner(), 1.0, time.time() - start_time
    
    def save_population(self, path = './training/checkpoint.json'):
        population = {}
        for player in self.players:
            population[player[3]] = {'weights': shaper.to_list(player[0]),
                                     'biases':  shaper.to_list(player[1]),
                                     'fitness': player[2]}
        json.dump(population, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'))

    
    def genetic_learning(self, show_games = False, save_checkpoint = True):
        total_n_of_games = self.N_PLAYERS * (self.N_KIDS + 1) * (self.N_PLAYERS * (self.N_KIDS + 1) - 1)

        #learn_start_time = time.time()

        for iteration in range(self.N_OF_ITERATIONS):
            current_players = []
            for player in self.players:
                current_players.append(copy.deepcopy(player))
                weights = player[0]
                biases = player[1]
                name = player[3]
                for _ in range(self.N_KIDS):
                    new_weights, new_biases = shaper.evolve(weights, biases)
                    current_players.append([new_weights, new_biases, 0, create_name(species_name = name)])
            
            current_game = 0
            times = []
            for player_index in range(len(current_players)):
                for enemy_index in range(len(current_players)):
                    if player_index != enemy_index:
                        iteration_start = time.time()
                        ########################################################
                        player_white = self.get_model()
                        player_white.set_weights(np.array(current_players[player_index][0]))
                        player_black = self.get_model()
                        player_black.set_weights(np.array(current_players[enemy_index][0]))

                        winner, fitness, _ = self.play([player_white, player_black], show_game=False)
                        if winner == 1:
                            current_players[player_index][2] += fitness
                        elif winner == 2:
                            current_players[enemy_index][2] += fitness
                        del player_white, player_black
                        current_game += 1
                        #########################################################
                        times.append((time.time() - iteration_start)/60)
                        approximated_wait_time = np.mean(np.array(times)) * (total_n_of_games - len(times))
                        printProgressBar(current_game,total_n_of_games, prefix = 'Progress:', suffix = 'Complete, approximated iteration time left: ' + "%.2f"%approximated_wait_time + ' min', length = 50)


            players_sorted = [player for player in sorted(current_players, key = lambda x: x[2], reverse=True)]      
            self.players = copy.deepcopy(players_sorted[:self.N_PLAYERS])
            print('iteration: ', iteration, 'winning fitness: ',self.players[0][2])
            for player_index in range(len(self.players)):
                print(self.players[player_index][2])
                self.players[player_index][2] = 0  

            if save_checkpoint:
                self.save_population()
                print('checkpoint saved')


        
          
training = AiModule(N_PLAYERS=10,N_KIDS = 4 ,N_OF_ITERATIONS=20)
training.genetic_learning()


    
    
    


            
        
        
