from Controller import *

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
        