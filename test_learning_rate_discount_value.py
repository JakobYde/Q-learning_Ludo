import ludopy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn
from ludopy import player
import qlearning

env = ludopy.Game()

EPISODES = 10000

ACTION_SPACE_SIZE = 4

INNER_STARS = [5, 18, 31, 44]
OUTER_STARS = [12, 25, 38, 51]

avg_window_size = 1000


ep_rewards = []
ep_won = []


rewards_table = {'star': 0.2, 'safe': 0.2, 'send_another_home': 0.2, 'send_self_home': -0.3, 'goal': 0.1,
                 'moved_into_goal_area': 0.2, 'out_of_start': 0.25, 'winner': 1, 'not_winner': -1}


def get_reward(moved_piece_previous_location, moved_piece_location, p_pieces, e_pieces, n_player_pieces_before, n_enemy_pieces_before):
    reward = 0
    if moved_piece_location in player.STAR_INDEXS:
        reward += rewards_table['star']
    elif moved_piece_location in player.GLOB_INDEXS or np.count_nonzero(p_pieces == moved_piece_location) > 1:
        reward += rewards_table['safe']
    elif moved_piece_location in player.HOME_AREAL_INDEXS\
            and moved_piece_previous_location not in player.HOME_AREAL_INDEXS:
        reward += rewards_table['moved_into_goal_area']
    elif moved_piece_location == player.GOAL_INDEX:
        reward += rewards_table['goal']
    elif moved_piece_location == player.START_INDEX:
        reward += rewards_table['out_of_start']

    n_p_pieces = len([piece for piece in p_pieces if piece != 0])
    n_e_pieces = len([piece for enemy in e_pieces for piece in enemy if piece != 0])

    if n_player_pieces_before > n_p_pieces:
        reward += rewards_table['send_self_home']
    if n_enemy_pieces_before > n_e_pieces:
        reward += rewards_table['send_another_home']

    another_has_won = False
    for enemy in enemy_pieces:
        if np.count_nonzero(enemy == player.GOAL_INDEX) == 4:
            reward += rewards_table['not_winner']
            another_has_won = True
            break
    if not another_has_won and np.count_nonzero(player_pieces == 59) == 4:
        reward += rewards_table['winner']

    return reward


def active_enemies_in_player_frame(enemy_pieces):

    enemy_offsets = [13, 26, 39]
    active_enemies = []

    for i, enemy in enumerate(enemy_pieces):
        for j, piece in enumerate(enemy):
            if piece not in [player.HOME_INDEX, player.GOAL_INDEX] and piece not in player.HOME_AREAL_INDEXS:
                if piece <= enemy_offsets[len(enemy_offsets) - 1 - i]:
                    active_enemies.append(piece + enemy_offsets[i])
                else:
                    active_enemies.append(piece - enemy_offsets[len(enemy_offsets) - 1 - i])

    return active_enemies


def get_state(player_pieces, enemy_pieces, dice, state_representation):
    if state_representation == 'eventbased':
        state = ''
        virtual_player = player.Player()
        for i in range(len(player_pieces)):
            virtual_player.set_pieces(player_pieces)
            virtual_player.move_piece(i, dice, enemy_pieces)
            piece_location = virtual_player.get_pieces()[i]
            temp_state = ['0'] * 8

            if piece_location in player.GLOB_INDEXS and piece_location in player_pieces:
                temp_state[0] = '1'
            if piece_location in [player.ENEMY_1_GLOB_INDX, player.ENEMY_2_GLOB_INDX, player.ENEMY_3_GLOB_INDX]:
                temp_state[1] = '1'
            if piece_location in player.STAR_INDEXS:
                temp_state[2] = '1'
            if piece_location in player.HOME_AREAL_INDEXS and player_pieces[i] not in player.HOME_AREAL_INDEXS:
                temp_state[3] = '1'
            if piece_location == player.START_INDEX:
                temp_state[4] = '1'
            if piece_location == player.GOAL_INDEX:
                temp_state[5] = '1'

            if player_pieces[i] == 0:
                if dice == 6:
                    piece_location = 1
                else:
                    piece_location = 0
            else:
                piece_location = player_pieces[i] + dice

            enemy_locations = active_enemies_in_player_frame(enemy_pieces)
            if piece_location in enemy_locations:
                if (np.count_nonzero(enemy_locations == piece_location) > 1
                    and piece_location != player.START_INDEX) \
                        or piece_location in np.append(player.GLOB_INDEXS, [player.ENEMY_1_GLOB_INDX,
                                                                            player.ENEMY_2_GLOB_INDX,
                                                                            player.ENEMY_3_GLOB_INDX]):
                    temp_state[6] = '1'
                else:
                    temp_state[7] = '1'

            if piece_location in player.STAR_INDEXS[:-1]:
                piece_location = player.STAR_INDEXS[np.where(player.STAR_INDEXS[:-1] == piece_location)[0][0] + 1]
                if piece_location in enemy_locations:
                    if np.count_nonzero(enemy_locations == piece_location) > 1:
                        temp_state[6] = '1'
                    else:
                        temp_state[7] = '1'

            state += ''.join(temp_state)
        return state

    elif state_representation == 'locations':
        state = ''
        virtual_player = player.Player()
        for i in range(len(player_pieces)):
            virtual_player.set_pieces(player_pieces)

            virtual_player.move_piece(i, dice, enemy_pieces)
            piece = virtual_player.get_pieces()[i]
            state += str(piece)
        return state

    else:
        state = ''
        virtual_player = player.Player()
        for i in range(len(player_pieces)):
            virtual_player.set_pieces(player_pieces)

            virtual_player.move_piece(i, dice, enemy_pieces)
            piece = virtual_player.get_pieces()[i]
            state += str(piece)

            enemy_locations = active_enemies_in_player_frame(enemy_pieces)
            temp_state = ['0', '0']

            if player_pieces[i] == 0:
                if dice == 6:
                    piece_location = 1
                else:
                    piece_location = 0
            else:
                piece_location = player_pieces[i] + dice

            if piece_location in enemy_locations:
                if (np.count_nonzero(enemy_locations == piece_location) > 1
                    and piece_location != player.START_INDEX) \
                        or piece_location in np.append(player.GLOB_INDEXS, [player.ENEMY_1_GLOB_INDX,
                                                                            player.ENEMY_2_GLOB_INDX,
                                                                            player.ENEMY_3_GLOB_INDX]):
                    temp_state[0] = '1'
                else:
                    temp_state[1] = '1'

            if piece_location in player.STAR_INDEXS[:-1]:
                piece_location = player.STAR_INDEXS[np.where(player.STAR_INDEXS[:-1] == piece_location)[0][0] + 1]
                if piece_location in enemy_locations:
                    if np.count_nonzero(enemy_locations == piece_location) > 1:
                        temp_state[0] = '1'
                    else:
                        temp_state[1] = '1'

            state += ''.join(temp_state)

        return state

state_representation = 'eventbased'


best_rewards = {}
best_win_rates = {}

learning_rates = [0.01, 0.05, 0.1]
discount_values = [0.7, 0.8, 0.9]

epsilon = 1
START_EPSILON_DECAYING = 1000
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

for learning_rate in learning_rates:
    for discount_value in discount_values:

        np.random.seed(10)

        q = qlearning.QLearning(learning_rate, discount_value, ACTION_SPACE_SIZE)

        for episode in range(EPISODES):
            episode_reward = 0

            if not episode % avg_window_size:
                print(episode)

            env.reset()
            done = False
            player_is_winner = False

            while not done:
                (dice, movable_pieces, player_pieces, enemy_pieces, player_is_winner, _), player_i = env.get_observation()

                if player_i == 0 and movable_pieces.size:

                    state = get_state(player_pieces, enemy_pieces, dice, state_representation)

                    if np.random.random() > epsilon:
                        action = movable_pieces[np.argmax(q.get_q(state)[movable_pieces])]
                    else:
                        action = movable_pieces[np.random.randint(0, movable_pieces.size)]

                    temp_player_pieces = player_pieces
                    n_player_pieces = len([piece for piece in player_pieces if piece != 0])
                    n_enemy_pieces = len([piece for enemy in enemy_pieces for piece in enemy if piece != 0])

                    _, movable_pieces, player_pieces, enemy_pieces, player_is_winner, done = env.answer_observation(action)

                    reward = get_reward(temp_player_pieces[action], player_pieces[action], player_pieces, enemy_pieces, n_player_pieces, n_enemy_pieces)

                    episode_reward += reward

                    new_state = get_state(player_pieces, enemy_pieces, dice, state_representation)

                    goal_condition = done

                    q.update_q_table(state, new_state, action, reward, done, goal_condition)

                elif movable_pieces.size:
                    action = movable_pieces[np.random.randint(0, len(movable_pieces))]
                    _, _, _, _, _, _ = env.answer_observation(action)
                else:
                    action = -1
                    _, _, _, _, _, _ = env.answer_observation(action)

            if player_is_winner:
                ep_won.append(100)
            else:
                ep_won.append(0)

            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value

            ep_rewards.append(episode_reward)

        mean_ep_won = bn.move_mean(ep_won, window=avg_window_size)
        mean_ep_rewards = bn.move_mean(ep_rewards, window=avg_window_size)

        best_rewards['learning_rate: {}, discount_value: {}'.format(learning_rate, discount_value)] = np.nanmax(mean_ep_rewards)
        best_win_rates['learning_rate: {}, discount_value: {}'.format(learning_rate, discount_value)] = np.nanmax(mean_ep_won)

for key in best_rewards:
    print(key, best_rewards[key])
    print(key, best_win_rates[key])

print('q_table contains: ', len(q.q_table), ' states')

'''
plt.plot(bn.move_mean(ep_won['locations'], window=avg_window_size), label="locations_avg_won")
plt.plot(bn.move_mean(ep_won['eventbased'], window=avg_window_size), label="events_avg_won")
plt.plot(bn.move_mean(ep_won['mix'], window=avg_window_size), label="mix_avg_won")
plt.legend(loc=4)
plt.xlabel('Episodes')
plt.ylabel('Win rate (%)')
plt.tight_layout()
plt.show()
plt.plot(bn.move_mean(ep_rewards['locations'], window=avg_window_size), label="locations_avg_reward")
plt.plot(bn.move_mean(ep_rewards['eventbased'], window=avg_window_size), label="events_avg_reward")
plt.plot(bn.move_mean(ep_rewards['mix'], window=avg_window_size), label="mix_avg_reward")
plt.legend(loc=4)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.tight_layout()
plt.show()
'''