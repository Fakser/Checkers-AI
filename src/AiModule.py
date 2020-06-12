from Controller import *
from Printing import *
from Numerical import *
from TextGenerator import *


class AiModule(object):
    """
    Class responsible for training the Neural Network, 
    storing all players and hyperparameters of the algorithm
    
    """
    def __init__(self, N_PLAYERS = 20, N_KIDS = 5, N_OF_ITERATIONS = 30):
        """
        Initializer of AIModule class
        @params:
            N_PLAYERS - Optional  : size of population (Int) 
            N_KIDS - Optional  : number of kids that each member has (Int)
            N_OF_ITERATIONS - Optional  : number of algorithm iterations (Int)
        """
        self.player_type = ['white', 'black']
        self.N_PLAYERS = N_PLAYERS 
        self.N_OF_ITERATIONS = N_OF_ITERATIONS
        self.N_KIDS = N_KIDS
        players = [np.array(self.get_model().get_weights()) for _ in range(self.N_PLAYERS)]
        for player_index in range(len(players)):
            players[player_index] = [copy.deepcopy(players[player_index]), shaper.get_biases(players[player_index]), 0, create_name(original=True)]
        self.players = players

    def get_model(self):
        """
        Method that returns Neural Network model of type Keras Model
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(64))
        model.add(tf.keras.layers.Dense(128, activation = 'relu'))
        model.add(tf.keras.layers.Dense(128, activation = 'relu'))
        model.add(tf.keras.layers.Dense(64, activation = 'relu'))
        model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'mse', optimizer = 'adadelta')
        return model
    
    def play(self, players, game_index = None, show_game = False):
        """
        Method that performs a game between two Neural Networks 
        and returns the winner with proper fitness
        @params:
            players - Requiered  : list of two keras models (list)
            game_index - Optional  : number of game in an iteration (Int)
            show_game - Optional  : If game should be printed to the console (Boolean)
        """
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
        """
        Saves current population to json file.
        Json architecture:
        {"spiecies name": {'weights': model weights,
                           'biases':  biases that are used to change weights,
                           'fitness': model fitness from previous iteration}
                            }
        @params:
            path - Optional  : path to the file which stores checkpoint (Boolean)
        """
        population = {}
        for player in self.players:
            population[player[3]] = {'weights': shaper.to_list(player[0]),
                                     'biases':  shaper.to_list(player[1]),
                                     'fitness': player[2]}
        json.dump(population, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'))

    
    def evolutionary_learning(self, show_games = False, save_checkpoint = True, tournament_type = 'tournament'):
        """
        Algorithm used for evolutionary learning of Neural Networks. 
        @params:
            save_checkpoint - Optional  : If checkpoint should be saved (Boolean)
            show_game - Optional  : If the game should be printed in real time (Boolean)
            tournament_type - Optional  : type of performing the algorithm. (String) OPTIONS: "allvsall" - each member of the population has to play with the rest. Good performance, resurce-greedy
                                                                                              "ttournament" - members are splitted into groups from which we choose bestt ones, then this process is repeated till one member is left, good performance, little less resurce-greedy
        """
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


        



    
    
    


            
        
        
