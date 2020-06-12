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
import gc

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

gc.enable()
#gc.set_debug(gc.DEBUG_LEAK)

from lorem.text import TextLorem

lorem = TextLorem(srange=(1,2))
used_names = []
def create_name(species_name = '', original = False):
    if original:
        while True:
            name = lorem.sentence().replace('.', '')
            if name not in used_names:
                return species_name + ' ' + name
    else:
        return species_name + ' ' + lorem.sentence().replace('.', '')

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

def greatest_divisor(number):
    max_div = 0
    for i in range(1, number - 1):
        if number % i == 0:
            max_div = i
    if max_div > 1:
        return max_div
    else:
        return number

def all_divisors(number):
    divisors = []
    for i in range(1, number - 1):
        if number % i == 0:
            divisors.append(i)
    if len(divisors) > 1:
        return divisors[1:]
    else:
        return [number]

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
        model.add(tf.keras.layers.Dense(128, activation = 'relu'))
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

    
    def genetic_learning(self, show_games = False, save_checkpoint = True, tournament_type = 'tournament'):
        total_n_of_games = self.N_PLAYERS * (self.N_KIDS + 1) * (self.N_PLAYERS * (self.N_KIDS + 1) - 1)

        #learn_start_time = time.time()

        for iteration in range(self.N_OF_ITERATIONS):
            current_players = []
            for player in self.players:
                current_players.append(copy.deepcopy(player))
                weights = copy.deepcopy(player[0])
                biases = copy.deepcopy(player[1])
                name = player[3]
                for _ in range(self.N_KIDS):
                    new_weights, new_biases = shaper.evolve(weights, biases)
                    current_players.append([new_weights, new_biases, 0, create_name(species_name = name)])
            
            current_game = 0
            times = []
            if tournament_type == 'allvsall':
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
                            random.shuffle(times)
                            times = times[:6]
                            approximated_wait_time = np.mean(np.array(times)) * (total_n_of_games - len(times))
                            printProgressBar(current_game,total_n_of_games, prefix = 'Progress:', suffix = 'Complete, approximated iteration time left: ' + "%.2f"%approximated_wait_time + ' min', length = 50)
                    #print('finished player' + str(player_index) +  'approximated iteration time left: ' + "%.2f"%approximated_wait_time + ' min')


            elif tournament_type == 'tournament':
                random.shuffle(current_players)
                group_size = min(all_divisors(len(current_players)))
                #n_groups = len(current_players)/group_size
                players_indexes = list(range(len(current_players)))
                print('tournament start')
                while group_size > 1:
                    group_index = 0
                    future_players = []
                    print([current_players[index][3] for index in players_indexes],players_indexes, group_size)
                    tournament_iteration = 0
                    for player_index in players_indexes:
                        for enemy_index in players_indexes[group_size * group_index: group_size * (group_index + 1)]:
                            if player_index != enemy_index:
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
                                print('group: ', group_index, ' white: ', player_index, ' black: ', enemy_index, 'winner: ', winner)


                        if (tournament_iteration + 1) % group_size == 0:
                            group_fitnesses = [current_players[index][2] for index in players_indexes[group_size * group_index: group_size * (group_index + 1)]]
                            group_indexes = [index for index in players_indexes[group_size * group_index: group_size * (group_index + 1)]]
                            future_players.append( group_indexes[group_fitnesses.index(max(group_fitnesses))])
                            print('group: ', group_index, 'winner: ', future_players[-1])
                            group_index += 1

                        tournament_iteration += 1

                    print('fitnesses: ', [current_players[index][2] for index in range(len(current_players))])
                    group_size = min(all_divisors(len(future_players)))
                    players_indexes =  copy.deepcopy(future_players)
                    n_groups = len(players_indexes)/group_size                   


            players_sorted = [player for player in sorted(current_players, key = lambda x: x[2], reverse=True)]      
            self.players = copy.deepcopy(players_sorted[:self.N_PLAYERS])
            del players_sorted
            del current_players
            print('iteration: ', iteration, 'winning fitness and spiecies: ', self.players[0][3], self.players[0][2])
            for player_index in range(len(self.players)):
                print('survives: ', self.players[player_index][3], self.players[player_index][2])
                self.players[player_index][2] = 0  

            if save_checkpoint:
                self.save_population()
                print('checkpoint saved')


        
          
training = AiModule(N_PLAYERS=4, N_KIDS = 2 ,N_OF_ITERATIONS=10)
training.genetic_learning(tournament_type='tournament')


    
    
    


            
        
        
