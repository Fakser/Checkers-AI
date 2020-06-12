from src.AiModule import AiModule

training = AiModule(N_PLAYERS=4, N_KIDS = 2 ,N_OF_ITERATIONS=10)
training.evolutionary_learning(tournament_type='tournament')