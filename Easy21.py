import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

ACTION_HIT = 0
ACTION_STAND = 1

N_ZERO = 1000
EPISODES = 50000

# Get a new card
def get_card(isFirst=False):
  # First cards are always black (+)
  card = np.random.randint(1, 11)
  if (isFirst):
    return card
  
  # Other cards are 2/3 black (+) and 1/3 red (-)
  if np.random.randint(1, 4) == 1:
    return -card

  return card

# Take in game step
def step(state, action):
  player_hand, dealer_hand = state

  # Player sticks -> dealer plays out turn and finish game
  if(action == ACTION_STAND):
    while dealer_hand <= 17 and dealer_hand >= 1:
      dealer_hand += get_card()

    # Player won states -> reward is 1
    if dealer_hand > 21 or player_hand > dealer_hand or dealer_hand < 1:
      return ((player_hand, dealer_hand), 1)
    
    # Game was drawn -> reward is 0
    if player_hand == dealer_hand:
      return ((player_hand, dealer_hand), 0)

    # Dealer won -> reward is -1
    return ((player_hand, dealer_hand), -1)

  # Player hits -> draw card for player
  player_hand += get_card()

  # Player went bust and loses -> reward is -1
  if player_hand > 21 or player_hand < 1:
    return ((player_hand, dealer_hand), -1)

  # Game continues
  return ((player_hand, dealer_hand), None)

# Greedy Policy
def epsilon_greedy_policy(Q, epsilon, numActions, state):
  actions = np.ones(numActions, dtype=float) * epsilon / numActions

  # Choose best action based on greedy policy (0 == hit, 1 == stand)
  # Play aggressively: If expected values are the same, choose hit
  best_action = 0 if Q[(state,0)] >= Q[(state,1)] else 1
  actions[best_action] += (1.0 - epsilon)
  
  return actions

# Monte Carlo Control 
def monte_carlo_control(numGames):
  # Count number of times state has been seen with default 0
  state_count = defaultdict(int)

  # Count number of times state has been seen and action taken with default 0
  state_action_count = defaultdict(int)

  # Generate Q function that monte_carlo_control returns (Action value function)
  Q = defaultdict(float)

  # Play out games in range given
  for games in range(0, numGames):
    # Keep track of all (state, action, reward) for game
    game_stats = []

    # Initialize game (always black (+) cards as first cards)
    player_hand = get_card(isFirst=True)
    dealer_hand = get_card(isFirst=True)
    state = (player_hand, dealer_hand)
    reward = None

    while True:
      # Increment number of times state has been seen
      state_count[state] += 1

      # Update/Init epsilon
      epsilon = N_ZERO / (N_ZERO + state_count[state])

      # Get probability matrix for actions and select best action (numAction == 2, hit or stand)
      probs = epsilon_greedy_policy(Q, epsilon, 2, state)
      action = np.random.choice(np.arange(len(probs)), p=probs)

      # Increment number of times state was seen and action taken
      state_action_count[(state, action)] += 1

      # Keep track of states seen and actions taken in within game
      game_stats.append((state, action))

      # Take step (perform action)
      next_state, reward = step(state, action)

      # Check if game finished
      if reward is not None:
        # Count final state
        state_count[next_state] += 1
        break

      state = next_state
    
    # Update Q function after game has finished
    for state_action in game_stats:
      alpha = 1 / state_action_count[state_action]
      Q[state_action] = Q[state_action] + alpha * (reward - Q[state_action])

  return Q


def TD_sarsa(numGames, lambda_arr, Qstar):
  # Each value is mean-squared error status for game
  ms_error_in_game = []

  # Play set number of games for each lambda
  for l in lambda_arr:
    # Count number of times state has been seen with default 0
    state_count = defaultdict(int)

    # Count number of times state has been seen and action taken with default 0
    state_action_count = defaultdict(int)

    # Generate Q function that monte_carlo_control returns (Action value function)
    Q = defaultdict(float)

    # Each lambda has an ms error array of length numGames.
    # Each value is mean-squared error when number of games played = index
    ms_error = []

    # Play games / episodes
    for game in range(0, numGames):
      # Init eligibility (count) for game with default 0
      E = defaultdict(int)

      # Initialize game
      state = (get_card(isFirst=True), get_card(isFirst=True))

      # Init epsilon
      epsilon = N_ZERO / (N_ZERO + state_count[state])

      # Init probability matrix and get first action
      probs = epsilon_greedy_policy(Q, epsilon, 2, state)
      action = np.random.choice(np.arange(len(probs)), p=probs)

      reward = 0

      # Play out game / episode
      while True:
        # Increment number of times state has been seen
        state_count[state] += 1

        # Increment number of times state was seen and action taken
        state_action_count[(state, action)] += 1

        # Take step (perform action)
        next_state, next_reward = step(state, action)

        # Update epsilon
        epsilon = N_ZERO / (N_ZERO + state_count[state])

        # Get Probability matrix for next action in next state and select action
        probs = epsilon_greedy_policy(Q, epsilon, 2, next_state)
        next_action = np.random.choice(np.arange(len(probs)), p=probs)

        # Update delta
        reward = 0 if not next_reward else next_reward
        delta = reward + Q[(next_state, next_action)] - Q[(state, action)]

        # Update eligibility count
        E[(state, action)] += 1

        # Update Q and E with all state actions seen
        for state_action in state_action_count:
          alpha = 1 / state_action_count[state_action]
          Q[state_action] = Q[state_action] + alpha * delta * E[state_action]
          E[state_action] = l * E[state_action]

        # Check if game is done
        if next_reward is not None:
          state_count [next_state] += 1
          break

        # Update state and action for next turn
        state = next_state
        action = next_action

      
      # Calculate mean-squared error 
      error_sum = 0.0
      for state_action in Qstar:
        error_sum += np.power((Q[state_action] - Qstar[state_action]), 2)

      # Save current mean-squared error
      ms_error.append(error_sum / len(Qstar))

    # Save all ms-error for this lambda
    ms_error_in_game.append(ms_error)

  # Returns array of arrays.
  # Each inner array represents ms errors for given lambda after each game
  return ms_error_in_game

#Change the Q from "dictonary" to "Array". Helper function to help plot Monte carlo control plot.
def Q2Array(Q):
    #Returns Array of arrays, index in array is player sum and index in inner array is dealer sum
    playerArray = []

     # Array has length 21 (player values from 1-21)
    for playerSum in range(1,22):
        dealerArray = []

        # Inner arrays has length 10 (Dealer values from 1-10)
        for dealerSum in range(1,11):
            #Get optimal action for (player,dealer) state (value in inner array is expected reward)
            dealerArray.append(max(Q[(playerSum,dealerSum), 0], Q[(playerSum,dealerSum), 1]))
        
        playerArray.append(dealerArray)
    
    return playerArray

# Plot heatmap for montecarlo
def PlotMonteCarloControl(results, episodes):
    title = str(episodes) + " Episodes"
    fig = sns.heatmap(np.flipud(results), cmap="YlGnBu", xticklabels=range(1,11),
            yticklabels=list(reversed(range(1,22))))
    fig.set_ylabel('player sum', fontsize=25)
    fig.set_xlabel('dealer showing', fontsize=25)
    fig.set_title(title, fontsize=25)

    plt.savefig('plots/mcControl.png')
    plt.close()

# Plot mean-squared error diff for all lambdas
def plotSarsaLambda(Qstar):
    lambdas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    games = 1000
    # Run sarsa, 1000 episodes for each lambda
    all_msErrors = TD_sarsa(games, lambdas, Qstar)
    
    # Plot mean-squared error diff for all lambdas
    finalError = []
    # all_msErrors is an array of arrays, each inner array represents ms errors for each some lambda
    for lambda_Error in all_msErrors:
        # Get last value in each inner array (ms error after a 1000 episodes)
        finalError.append(lambda_Error[len(lambda_Error) - 1])

    plt.plot(lambdas, finalError)
    plt.xlabel('Lambda')
    plt.ylabel('Mean square error')

    plt.savefig('plots/MSerror.png')
    plt.close()

    # Plot learning curve for lambda=1 and lambda=0
    plt.plot(all_msErrors[0], label='Lambda 0.0')
    plt.plot(all_msErrors[10], label='Lambda 1.0')
    plt.xlabel('Episodes')
    plt.ylabel('Mean square error')
    plt.legend()

    plt.savefig('plots/learningCurve.png')
    plt.close()

def main():
  # Play Monte Carlo control games, returns Q function
  mcQ = monte_carlo_control(EPISODES)

  # Change monte carlo Q to array to plot, and then plot heatmap
  Qarray = Q2Array(mcQ)
  PlotMonteCarloControl(Qarray, EPISODES)

  # Use mcQ as Q* and plot mean-squared error for lambda 0, 0.1 , ... , 1 for a 1000 episodes each
  # Use mcQ as Q* and plot the learning curve agains number of episodes for lambda = 0.0 and lambda = 1.0
  plotSarsaLambda(mcQ) 

if __name__ == '__main__':
	main()

        
















