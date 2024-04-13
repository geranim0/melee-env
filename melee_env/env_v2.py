from melee_env.dconfig import DolphinConfig
import melee
from melee import enums
import numpy as np
import sys
import time
from pathlib import Path
import copy
from melee_env import sam_utils
import random
from melee_env.agents.basic import *
import gymnasium as gym
from datetime import datetime

class MeleeEnv_v2(gym.Env):
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, 
        iso_path,
        slippi_game_path,
        players,
        fast_forward=False, 
        blocking_input=True,
        ai_starts_game=True,
        shuffle_controllers_after_each_game=True,
        randomize_stage = True,
        randomize_character = True,
        num_players = 2,
        max_match_steps = 60*60*8,
        action_repeat = 12,
        env_num = "0",
        slippi_port = "51441",
        seed = 69):

        random.seed(seed)
        self.d = DolphinConfig(slippi_game_path, env_num)
        self.d.set_ff(fast_forward)

        self.iso_path = Path(iso_path).resolve()
        self.players = players

        self._logical_actions_to_controller_actions_fn = sam_utils.logical_v1_to_libmelee_inputs
        self._gamestate_to_obs_space_fn = self._gamestate_to_obs_space_fn_v1

        # inform other players of other players
        # for player in self.players:
        #     player.set_player_keys(len(self.players))
        
        if len(self.players) == 2:
            self.d.set_center_p2_hud(True)
        else:
            self.d.set_center_p2_hud(False)

        self.blocking_input = blocking_input
        self.ai_starts_game = ai_starts_game
        self.slippi_port = slippi_port

        self._shuffle_controllers_after_each_game = shuffle_controllers_after_each_game
        self._randomize_stage = randomize_stage
        self._randomize_character = randomize_character
        self._num_players = num_players

        self.gamestate = None
        self.previous_gamestate = None
        self.env_is_started = False

        self._friendly_ports = None
        self._enemy_ports = None

        self._dead_ports = {}
        self.removethis = 0

        self.action_space = self._get_action_space()
        self.observation_space = self._get_obervation_space()

        self._current_match_steps = 0
        self._max_match_steps = max_match_steps
        self._action_repeat = action_repeat
        self.console = None


    @staticmethod
    def get_action_space_v1(num_players):
        # left, right, up, down and diags + nothing = 9 
        # A, B, full R, = 3
        # => 12
        num_joystick_positions = 9

        if num_players == 2:
            return gym.spaces.MultiDiscrete([num_joystick_positions, 2, 2, 2])
        elif num_players == 4:
            return gym.spaces.MultiDiscrete([num_joystick_positions, num_joystick_positions, 2, 2, 2, 2, 2, 2])
        else:
            return NotImplementedError("num players must be 2 or 4. Got: " + str(num_players))
        
    @staticmethod
    def get_action_space_v2(num_players):
        # left, right, up, down and diags + nothing = 9 
        # A, B, full R, = 3
        # => 12
        num_joystick_horizontal_positions = 3
        num_joystick_vertical_positions = 3

        if num_players == 2:
            return gym.spaces.MultiDiscrete([num_joystick_horizontal_positions, num_joystick_vertical_positions, 2, 2, 2])
        elif num_players == 4:
            return NotImplementedError("num players must be 2. Got: " + str(num_players))
        else:
            return NotImplementedError("num players must be 2 or 4. Got: " + str(num_players))
        
    
    @staticmethod
    def get_observation_space_v1(num_players):
        # action (enum.Action): one hot (385 bits)
        # action_frame (int):
        # character (enum.Character): 
        # facing (bool): 
        # hitstun_frames_left (int): 
        # invulnerability_left (int): 
        # jumps_left (int): 
        # off_stage (bool):
        # on_ground (bool): 
        # percent (int):
        # position (float, float): 
        # shield_strength (float max 60): 
        # speed_air_x_self (float): 
        # speed_ground_x_self(float): 
        # speed_x_attack (float): 
        # speed_y_attack (float):
        # speed_y_self (float): 
        # stock (int): 
        # ----- for later ---------
        # stage, projectiles,  

        # one hot encoded obs
        action_len = len(list(melee.enums.Action)) # 385
        character_len = len(list(melee.enums.Character))
        facing_len = 1
        offstage_len = 1
        on_ground_len = 1

        # todo: try multibinary and integer type and see diff. is this wasteful?
        # action, character, facing, off_stage, on_ground
        one_hot_obs = gym.spaces.Box(low=0, high=1, shape=(action_len + character_len + facing_len + offstage_len + on_ground_len,), dtype=np.float32)

        # int obs
        # action_frame, hitstun_frames_left, invulnerability_left, jumps_left, percent, stock
        int_obs = gym.spaces.Box(
            low=0,
            high=999,
            shape=(6,),
            dtype=np.float32)

        #float obs
        # position_x, position_y, shield strength, speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
        float_obs = gym.spaces.Box(
            low=-np.Infinity,
            high=np.Infinity,
            shape=(8,),
            dtype=np.float32)
        

        if num_players == 2:
            all_obs_space = gym.spaces.Dict({
                'one_hot_p1': one_hot_obs,
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'one_hot_p2': one_hot_obs,
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,
            })
        elif num_players == 4:
            all_obs_space = gym.spaces.Dict({
                'one_hot_p1': one_hot_obs,
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'one_hot_p2': one_hot_obs,
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,
                
                'one_hot_p3': one_hot_obs,
                'int_obs_p3': int_obs,
                'float_obs_p3': float_obs,
                
                'one_hot_p4': one_hot_obs,
                'int_obs_p4': int_obs,
                'float_obs_p4': float_obs,
            })
        else:
            raise NotImplementedError('num_players must be 2 or 4')

        return all_obs_space
    

    @staticmethod
    def get_observation_space_v2(num_players):
        # action (enum.Action): one hot (385 bits)
        # action_frame (int):
        # character (enum.Character): 
        # facing (bool): 
        # hitstun_frames_left (int): 
        # invulnerability_left (int): 
        # jumps_left (int): 
        # off_stage (bool):
        # on_ground (bool): 
        # percent (int):
        # position (float, float): 
        # shield_strength (float max 60): 
        # speed_air_x_self (float): 
        # speed_ground_x_self(float): 
        # speed_x_attack (float): 
        # speed_y_attack (float):
        # speed_y_self (float): 
        # stock (int): 
        # ----- for later ---------
        # stage, projectiles,  

        # one hot encoded obs
        #stage = 1
        action_len = len(list(melee.enums.Action)) # 385
        character_len = len(list(melee.enums.Character))
        facing_len = 1
        offstage_len = 1
        on_ground_len = 1

        # todo: try multibinary and integer type and see diff. is this wasteful?
        # action, character, facing, off_stage, on_ground
        one_hot_obs = gym.spaces.Box(low=0, high=1, shape=(action_len + character_len + facing_len + offstage_len + on_ground_len,), dtype=np.float32)

        # int obs
        # action_frame, hitstun_frames_left, invulnerability_left, jumps_left, percent, stock
        int_obs = gym.spaces.Box(
            low=0,
            high=999,
            shape=(6,),
            dtype=np.float32)

        #float obs
        # position_x, position_y, shield strength, speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
        float_obs = gym.spaces.Box(
            low=-np.Infinity,
            high=np.Infinity,
            shape=(8,),
            dtype=np.float32)
        

        if num_players == 2:
            all_obs_space = gym.spaces.Dict({
                'one_hot_p1': one_hot_obs,
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'one_hot_p2': one_hot_obs,
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,
            })
        elif num_players == 4:
            all_obs_space = gym.spaces.Dict({
                'one_hot_p1': one_hot_obs,
                'int_obs_p1': int_obs,
                'float_obs_p1': float_obs,
                
                'one_hot_p2': one_hot_obs,
                'int_obs_p2': int_obs,
                'float_obs_p2': float_obs,
                
                'one_hot_p3': one_hot_obs,
                'int_obs_p3': int_obs,
                'float_obs_p3': float_obs,
                
                'one_hot_p4': one_hot_obs,
                'int_obs_p4': int_obs,
                'float_obs_p4': float_obs,
            })
        else:
            raise NotImplementedError('num_players must be 2 or 4')

        return all_obs_space

    def _get_action_space(self):
        return MeleeEnv_v2.get_action_space_v1(self._num_players)

    def _get_obervation_space(self):
        return MeleeEnv_v2.get_observation_space_v1(self._num_players)
    
    @staticmethod
    def gamestate_to_obs_space_v1(gamestate, friendly_ports, enemy_ports):
        # one hot
        action = {port: gamestate.players[port].action for port in gamestate.players.keys()}
        character = {port: gamestate.players[port].character for port in gamestate.players.keys()}
        facing = {port: gamestate.players[port].facing for port in gamestate.players.keys()}
        off_stage = {port: gamestate.players[port].off_stage for port in gamestate.players.keys()}
        on_ground = {port: gamestate.players[port].on_ground for port in gamestate.players.keys()}

        # int obs
        action_frame = {port: gamestate.players[port].action_frame for port in gamestate.players.keys()}
        hitstun_frames_left = {port: gamestate.players[port].hitstun_frames_left for port in gamestate.players.keys()}
        invulnerability_left = {port: gamestate.players[port].invulnerability_left for port in gamestate.players.keys()}
        jumps_left = {port: gamestate.players[port].jumps_left for port in gamestate.players.keys()}
        percent = {port: gamestate.players[port].percent for port in gamestate.players.keys()}
        stock = {port: gamestate.players[port].stock for port in gamestate.players.keys()}

        # float obs
        position = {port: gamestate.players[port].position for port in gamestate.players.keys()}
        shield_strength = {port: gamestate.players[port].shield_strength for port in gamestate.players.keys()}
        speed_air_x_self = {port: gamestate.players[port].speed_air_x_self for port in gamestate.players.keys()}
        speed_ground_x_self = {port: gamestate.players[port].speed_ground_x_self for port in gamestate.players.keys()}
        speed_x_attack = {port: gamestate.players[port].speed_x_attack for port in gamestate.players.keys()}
        speed_y_attack = {port: gamestate.players[port].speed_y_attack for port in gamestate.players.keys()}
        speed_y_self = {port: gamestate.players[port].speed_y_self for port in gamestate.players.keys()}

        current_player_index = 1
        obs = {}
        for port in np.concatenate((friendly_ports, enemy_ports), axis=None):
            
            # one hot
            one_hot_dict_idx = "one_hot_p" + str(current_player_index)

            action_one_hot = np.zeros(len(list(melee.enums.Action)))
            action_one_hot[action[port].value % len(list(melee.enums.Action))] = 1

            if action[port].value >= len(list(melee.enums.Action)):
                print('Error: illegal action: ' + str(action[port].value))

            character_one_hot = np.zeros(len(list(melee.enums.Character)))
            character_one_hot[character[port].value % len(list(melee.enums.Character))] = 1

            if character[port].value >= len(list(melee.enums.Character)):
                print('Error: illegal character: ' + str(action[port].value))

            all_one_hots = np.concatenate((action_one_hot, character_one_hot, facing[port], off_stage[port], on_ground[port]), axis=None).astype(np.float32)
            obs[one_hot_dict_idx] = all_one_hots

            # int
            int_dict_idx = "int_obs_p" + str(current_player_index)
            all_ints = np.concatenate(
                (
                    action_frame[port],
                    hitstun_frames_left[port],
                    invulnerability_left[port],
                    jumps_left[port],
                    percent[port],
                    stock[port]
                ), axis=None).astype(np.float32)
            obs[int_dict_idx] = all_ints

            # floats
            float_dict_idx = "float_obs_p" + str(current_player_index)
            all_floats = np.concatenate(
                (
                    position[port].x,
                    position[port].y,
                    shield_strength[port],
                    speed_air_x_self[port],
                    speed_ground_x_self[port],
                    speed_x_attack[port],
                    speed_y_attack[port],
                    speed_y_self[port]
                ), axis=None).astype(np.float32)
            obs[float_dict_idx] = all_floats

            current_player_index += 1
        
        return obs

    def _gamestate_to_obs_space_fn_v1(self, gamestate):        
        return MeleeEnv_v2.gamestate_to_obs_space_v1(gamestate, self._friendly_ports, self._enemy_ports)

    def start(self):
        if sys.platform == "linux":
            dolphin_home_path = str(self.d.slippi_home)+"/"
        elif sys.platform == "win32":
            dolphin_home_path = None

        self.console = melee.Console(
            path=str(self.d.slippi_bin_path),
            dolphin_home_path=dolphin_home_path,
            blocking_input=self.blocking_input,
            tmp_home_directory=True,
            gfx_backend='Vulkan',
            slippi_port=self.slippi_port)

        # print(self.console.dolphin_home_path)  # add to logging later
        # Configure Dolphin for the correct controller setup, add controllers
        human_detected = False

        for i in range(len(self.players)):
            curr_player = self.players[i]
            if curr_player.agent_type == agent_type.HMN:
                self.d.set_controller_type(i+1, enums.ControllerType.GCN_ADAPTER)
                curr_player.controller = melee.Controller(console=self.console, port=i+1, type=melee.ControllerType.GCN_ADAPTER)
                human_detected = True
            elif curr_player.agent_type in [agent_type.step_controlled_AI, agent_type.enemy_controlled_AI, agent_type.CPU, agent_type.HARDCODED]:
                self.d.set_controller_type(i+1, enums.ControllerType.STANDARD)
                curr_player.controller = melee.Controller(console=self.console, port=i+1)
                self.menu_control_agent = curr_player
            else:  # no player
                self.d.set_controller_type(i+1, enums.ControllerType.UNPLUGGED)
            
        if self.ai_starts_game and not human_detected:
            self.ai_press_start = True

        else:
            self.ai_press_start = False  # don't let ai press start without the human player joining in. 

        #if self.ai_starts_game and self.ai_press_start:
            #self.players[self.menu_control_agent].press_start = True

        self.console.run(iso_path=self.iso_path)
        self.console.connect()

        [player.controller.connect() for player in self.players if player is not None]

        self.gamestate = self.console.step()
        self.env_is_started = True
 
    # so the agent learns to play each port. todo: do with ports, no controller
    def _shuffle_controllers(self):
        remaining_controllers = [player.controller for player in self.players]

        for player in self.players:
            chosen_controller = random.choice(remaining_controllers)
            player.controller = chosen_controller
            remaining_controllers.remove(chosen_controller)

    def _choose_stage(self):
        if self._randomize_stage:
            stage =  random.choice([
                melee.enums.Stage.FINAL_DESTINATION,
                melee.enums.Stage.BATTLEFIELD,
                melee.enums.Stage.POKEMON_STADIUM,
                melee.enums.Stage.DREAMLAND,
                melee.enums.Stage.FOUNTAIN_OF_DREAMS,
                melee.enums.Stage.YOSHIS_STORY])
        else:
            stage = melee.enums.Stage.BATTLEFIELD
        return stage

    def _randomize_characters(self):
        for player in self.players:
            #if type(player) is sam_ai:
            player.character = random.choice([
                melee.enums.Character.BOWSER,
                melee.enums.Character.CPTFALCON,
                melee.enums.Character.DK,
                melee.enums.Character.DOC,
                melee.enums.Character.FALCO,
                melee.enums.Character.FOX,
                melee.enums.Character.GAMEANDWATCH,
                melee.enums.Character.GANONDORF,
                melee.enums.Character.JIGGLYPUFF,
                #melee.enums.Character.KIRBY, has illegal actions 65535 and 396
                melee.enums.Character.LINK,
                melee.enums.Character.LUIGI,
                melee.enums.Character.MARIO,
                melee.enums.Character.MARTH,
                melee.enums.Character.MEWTWO,
                melee.enums.Character.NESS,
                melee.enums.Character.PEACH,
                melee.enums.Character.PICHU,
                melee.enums.Character.PIKACHU,
                melee.enums.Character.POPO,
                melee.enums.Character.ROY,
                melee.enums.Character.SAMUS,
                #melee.enums.Character.SHEIK, it makes it bug
                melee.enums.Character.YLINK,
                melee.enums.Character.YOSHI,
                melee.enums.Character.ZELDA])
            if player.agent_type == agent_type.step_controlled_AI:
                print('char = ' + str(player.character) + 'at: ' + str(datetime.now().isoformat()))

    def _populate_friendly_enemy_ports(self):
        self._friendly_ports = []
        self._enemy_ports = []
        
        for player in self.players:
            if player.agent_type == agent_type.step_controlled_AI: #todo: replace all othose with an enum
                self._friendly_ports.append(player.controller.port)
            else:
                self._enemy_ports.append(player.controller.port)

    def select_character(self):
        
        all_players_press_nothing(self.players)

        cpu_players = [player for player in self.players if player.agent_type == agent_type.CPU]
        other_players = [player for player in self.players if player not in cpu_players]


        for player in np.concatenate((cpu_players, other_players), axis=None):
            while not melee.MenuHelper.choose_character(
                            character=player.character,
                            gamestate=self.gamestate,
                            controller=player.controller,
                            costume=0, # todo: random this
                            swag=False,
                            start=False,
                            cpu_level=player.lvl if player.agent_type == agent_type.CPU else 0):
                #print('player port: ' + str(player.controller.port) + ' just chose ' + str(player.character))
                self.gamestate = self.console.step()
        
        current_frame = 0
        while self.gamestate.menu_state != melee.Menu.STAGE_SELECT:
            if current_frame % 2 == 0:
                all_players_press_nothing(self.players)
                self.gamestate = self.console.step()
                self.players[0].controller.press_button(melee.enums.Button.BUTTON_START)
                self.players[0].controller.flush()

            self.gamestate = self.console.step()
            self.players[0].controller.release_all()
            current_frame += 1
        

    def setup(self):
        self.previous_gamestate = None
        self._current_match_steps = 0

        for player in self.players:
            player.defeated = False

        if self._shuffle_controllers_after_each_game == True:
            self._shuffle_controllers()
        
        if self._randomize_character == True:
            self._randomize_characters()
        
        chosen_stage = self._choose_stage()

        self._populate_friendly_enemy_ports()

        for i in range(1, 5):
            self._dead_ports[i] = False
            
        while True:
            #all_players_press_nothing(self.players)
            self.gamestate = self.console.step()

            if self.gamestate.menu_state is melee.Menu.CHARACTER_SELECT:
                self.select_character()
                #for player in self.players:
                #    if player.agent_type in ["AI", "HardCoded"] :
                #        melee.MenuHelper.choose_character(
                #            character=player.character,
                #            gamestate=self.gamestate,
                #            controller=player.controller,
                #            costume=i,
                #            swag=False,
                #            start=False)
                #            #start=self.players[i].press_start)
                #    if player.agent_type == "CPU":
                #        melee.MenuHelper.choose_character(
                #            character=player.character,
                #            gamestate=self.gamestate,
                #            controller=player.controller,
                #            costume=i,
                #            swag=False,
                #            cpu_level=player.lvl,
                #            start= False)
                #            #start=self.players[i].press_start)
                #melee.MenuHelper.menu_helper_simple(self.gamestate, )

            elif self.gamestate.menu_state is melee.Menu.STAGE_SELECT:
                all_players_press_nothing(self.players)
                #print('choosin stage with controller port: ' + str(self.menu_control_agent.controller.port))
                melee.MenuHelper.choose_stage(
                    stage=chosen_stage,
                    gamestate=self.gamestate,
                    controller=self.menu_control_agent.controller)

            elif self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return self.gamestate, False  # game is not done on start
                
            else:
                melee.MenuHelper.choose_versus_mode(self.gamestate, self.menu_control_agent.controller)
    
    def get_stocks(self, gamestate):
        stocks = [gamestate.players[i].stock for i in list(gamestate.players.keys())]
        return np.array([stocks]).T  # players x 1
    
    # returns list of [port, stocks]
    def get_stocks_v2(self, gamestate):
        return {port : int(gamestate.players[port].stock) for port in gamestate.players.keys()}
    
    def get_characters(self):
        return {port : self.gamestate.players[port].character for port in self.gamestate.players.keys()}
  
    def get_actions(self, gamestate):
        actions = [gamestate.players[i].action.value for i in list(gamestate.players.keys())]
        action_frames = [gamestate.players[i].action_frame for i in list(gamestate.players.keys())]
        hitstun_frames_left = [gamestate.players[i].hitstun_frames_left for i in list(gamestate.players.keys())]
        
        return np.array([actions, action_frames, hitstun_frames_left]).T # players x 3
    
    def get_positions(self, gamestate):
        x_positions = [gamestate.players[i].position.x for i in list(gamestate.players.keys())]
        y_positions = [gamestate.players[i].position.y for i in list(gamestate.players.keys())]

        return np.array([x_positions, y_positions]).T  # players x 2
    
    def get_damages(self, gamestate):
        damages = [gamestate.players[i].percent for i in list(gamestate.players.keys())]
        return np.array([damages]).T  # players x 1
    
    # returns list of [port, damage]
    def get_damages_v2(self, gamestate):
        return {port: int(gamestate.players[port].percent) for port in gamestate.players.keys()}

    # returns rewards for all players
    def calculate_rewards_v1(self, previous_gamestate, current_gamestate):

        if not previous_gamestate:
            return [0 for i in list(current_gamestate.players.keys())]

        previous_stocks = self.get_stocks(previous_gamestate)
        current_stocks = self.get_stocks(current_gamestate)

        stock_differential = [k[0] for k in previous_stocks - current_stocks]
        #if stock_differential[0] or stock_differential[1]:
            #print('stock taken')
            #print(stock_differential)

        previous_damages = self.get_damages(previous_gamestate)
        current_damages = self.get_damages(current_gamestate)

        damages_differential = [max(0, k[0]) for k in current_damages - previous_damages]

        #if damages_differential[0] or damages_differential[1]:
            #print('damage taken')
            #print(damages_differential)

        # its actually lost hte game tho
        won_the_game = [bool(current_stocks[i] == 0) for i in [0,1]] # todo: fix for 4 player game

        #if won_the_game[0] or won_the_game[1]:
            #print('game over')
            #print(won_the_game)

        stock_multiplier = 200
        damage_multiplier = 1
        win_multiplier = 0

        p0_rewards = (stock_differential[1] - stock_differential[0]) * stock_multiplier\
            + (damages_differential[1] - damages_differential[0]) * damage_multiplier\
            + (won_the_game[1] - won_the_game[0]) * win_multiplier # win
        
        p1_rewards = -p0_rewards


        rewards = [p0_rewards, p1_rewards]
        
        #if rewards[0] or rewards[1]:
            #print(rewards)
        
        #if rewards[0] == 1:
            #print(stock_differential)
            #print(damages_differential)
            #print(won_the_game)

        return rewards
    
    def calculate_rewards_v3(self, friendly_ports, enemy_ports, previous_gamestate, current_gamestate):
        # todo: take as arg
        stock_multiplier = 200
        damage_multiplier = 1

        if not previous_gamestate:
            return 0

        previous_stocks = self.get_stocks_v2(previous_gamestate)
        current_stocks = self.get_stocks_v2(current_gamestate)

        stock_differential = {port: current_stocks[port] - previous_stocks[port] for port in previous_stocks}

        #if done and (stock_differential[port] == 0 for port in stock_differential):
            #return 0 

        for port in [port for port in stock_differential if stock_differential[port] != 0]:
            self._dead_ports[port] = True

        previous_damages = self.get_damages_v2(previous_gamestate)
        current_damages = self.get_damages_v2(current_gamestate)

        damages_differential = {}
        for port in previous_damages:
            damages_differential[port] = current_damages[port] - previous_damages[port]
            
            if damages_differential[port] < 0 and self._dead_ports[port] == True:
                damages_differential[port] = 0
                self._dead_ports[port] = False
            

        for port in [port for port in damages_differential if damages_differential[port] != 0 and self._dead_ports[port] == True]:
            self._dead_ports[port] = False

        won_the_game = all(current_stocks[port] == 0 for port in enemy_ports) and not all(current_stocks[port] == 0 for port in friendly_ports)

        stock_rewards = (sum(stock_differential[port] for port in friendly_ports) \
                        - sum(stock_differential[port] for port in enemy_ports)) \
                        * stock_multiplier
    

        damage_rewards = (sum(damages_differential[port] for port in enemy_ports) \
            - sum(damages_differential[port] for port in friendly_ports)) \
            * damage_multiplier

        rewards = stock_rewards + damage_rewards
        return rewards

    # returns rewards only for the agent being trained
    def calculate_rewards_v2(self, friendly_ports, enemy_ports, previous_gamestate, current_gamestate):
        # todo: take as arg
        stock_multiplier = 1000
        damage_multiplier = 1
        time_tick_neg_reward = -1
        win_multiplier = 0

        if not previous_gamestate:
            return 0

        previous_stocks = self.get_stocks_v2(previous_gamestate)
        current_stocks = self.get_stocks_v2(current_gamestate)

        stock_differential = {port: current_stocks[port] - previous_stocks[port] for port in previous_stocks}

        #if done and (stock_differential[port] == 0 for port in stock_differential):
            #return 0 

        for port in [port for port in stock_differential if stock_differential[port] != 0]:
            self._dead_ports[port] = True

        previous_damages = self.get_damages_v2(previous_gamestate)
        current_damages = self.get_damages_v2(current_gamestate)

        damages_differential = {}
        for port in previous_damages:
            damages_differential[port] = current_damages[port] - previous_damages[port]
            
            if damages_differential[port] < 0 and self._dead_ports[port] == True:
                damages_differential[port] = 0
                self._dead_ports[port] = False
            

        for port in [port for port in damages_differential if damages_differential[port] != 0 and self._dead_ports[port] == True]:
            self._dead_ports[port] = False

        won_the_game = all(current_stocks[port] == 0 for port in enemy_ports) and not all(current_stocks[port] == 0 for port in friendly_ports)

        stock_rewards = (sum(stock_differential[port] for port in friendly_ports) \
                        - sum(stock_differential[port] for port in enemy_ports)) \
                        * stock_multiplier
    

        damage_rewards = (sum(damages_differential[port] for port in enemy_ports) \
            - sum(damages_differential[port] for port in friendly_ports)) \
            * damage_multiplier

        rewards = time_tick_neg_reward + stock_rewards + damage_rewards
        return rewards
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.env_is_started:
            self.start()

        while self.gamestate.menu_state == melee.Menu.IN_GAME:
            #print('reset, trying to jump offstage')
            for player in self.players:
                player.controller.release_all()
                player.controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, 1, 0.5)
                player.controller.flush()
            self.gamestate = self.console.step()


        # reset players (maybe put this in reset in base class?)        
        all_players_press_nothing(self.players)
        for player in self.players:
            player.reset()

        
        obs, done = self.setup()
        return self._gamestate_to_obs_space_fn(obs), self._get_info() # obs, info

    def _get_info(self):
        return {}

    # todo: fix for 4 players and make it cleaner
    def _is_done(self):
        stocks = self.get_stocks_v2(self.gamestate)

        friendly_stocks = [stocks[port] for port in self._friendly_ports]
        if all(stocks == 0 for stocks in friendly_stocks):
            #print('all friendly stocks == 0, resetting')
            return True
        
        enemy_stocks = [stocks[port] for port in self._enemy_ports]
        if all(stocks == 0 for stocks in enemy_stocks):
            #print('all enemy stocks == 0, resetting')
            return True
        
        #if self.gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
        #    return True

        return False
    
    def _gen_step(self, agent_to_logical_actions_fn):
        def step(agent_actions):
            logical_actions = agent_to_logical_actions_fn(agent_actions)
            return self.step_logical_v2(logical_actions)
        return step

    def render(self):
        pass

    def seed(self, seed=None):
        pass

    def step(self, actions):
        return self.step_v4(actions)

    def step_v4(self, raw_step_controlled_agent_actions):
        done = self._is_done()
        rewards = 0
        truncated = None
        infos = {}

        for i in range(0, self._action_repeat):
            if self.gamestate.menu_state == melee.Menu.IN_GAME and not done:
                
                if self._current_match_steps < self._max_match_steps:
                    for player in self.players:
                        if player.agent_type == agent_type.step_controlled_AI:
                            logical_actions = player.raw_agent_actions_to_logical_fn(raw_step_controlled_agent_actions)
                            controller_actions = player.logical_to_controller_fn(logical_actions, i)
                            this_agent_controller = get_agent_controller(player)
                            execute_actions(this_agent_controller, controller_actions)
                        elif player.agent_type == agent_type.enemy_controlled_AI:
                            if (i == 0):
                                obs = player.gamestate_to_observation_fn(self.gamestate, self._enemy_ports, self._friendly_ports)
                                raw_actions = player.observation_to_raw_inputs_fn(obs)
                            
                            logical_actions = player.raw_agent_actions_to_logical_fn(raw_actions)
                            controller_actions = player.logical_to_controller_fn(logical_actions, i)
                            this_agent_controller = get_agent_controller(player)
                            execute_actions(this_agent_controller, controller_actions)
                        else:
                            player.act(self.gamestate)

                
                else:
                    all_players_press_nothing(self.players)
                    return self._gamestate_to_obs_space_fn(self.gamestate), 0, True, True, infos


                self.gamestate = self.console.step()
                self._current_match_steps += 1

                done = self._is_done()

                rewards += self.calculate_rewards_v3(self._friendly_ports, self._enemy_ports, self.previous_gamestate, self.gamestate)
        
                self.previous_gamestate = self.gamestate

                if done:
                    all_players_press_nothing(self.players) # if A is pressed at the end, skips char select
                    break

        return self._gamestate_to_obs_space_fn(self.gamestate), rewards, done, truncated, infos



    def step_v3(self, raw_step_controlled_agent_actions):
        done = self._is_done()
        rewards = 0
        truncated = None
        infos = {}

        for i in range(0, self._action_repeat):
            if self.gamestate.menu_state == melee.Menu.IN_GAME and not done:
                
                if self._current_match_steps < self._max_match_steps:
                    for player in self.players:
                        if player.agent_type == agent_type.step_controlled_AI:
                            logical_actions = player.raw_agent_actions_to_logical_fn(raw_step_controlled_agent_actions)
                            controller_actions = player.logical_to_controller_fn(logical_actions, i)
                            this_agent_controller = get_agent_controller(player)
                            execute_actions(this_agent_controller, controller_actions)
                        elif player.agent_type == agent_type.enemy_controlled_AI:
                            obs = player.gamestate_to_observation_fn(self.gamestate, self._enemy_ports, self._friendly_ports)
                            raw_actions = player.observation_to_raw_inputs_fn(obs)
                            logical_actions = player.raw_agent_actions_to_logical_fn(raw_actions)
                            controller_actions = player.logical_to_controller_fn(logical_actions, i)
                            this_agent_controller = get_agent_controller(player)
                            execute_actions(this_agent_controller, controller_actions)
                        else:
                            player.act(self.gamestate)

                
                else:
                    all_players_press_nothing(self.players)
                    return self._gamestate_to_obs_space_fn(self.gamestate), 0, True, True, infos


                self.gamestate = self.console.step()
                self._current_match_steps += 1

                done = self._is_done()

                rewards += self.calculate_rewards_v2(self._friendly_ports, self._enemy_ports, self.previous_gamestate, self.gamestate)
        
                self.previous_gamestate = self.gamestate

                if done:
                    all_players_press_nothing(self.players) # if A is pressed at the end, skips char select
                    break

        return self._gamestate_to_obs_space_fn(self.gamestate), rewards, done, truncated, infos
    
    # supports AI enemies
    def step_logical_v2(self, logical_actions):
        done = self._is_done()
        rewards = 0
        truncated = None
        infos = {}

        for i in range(0, self._action_repeat):
            if self.gamestate.menu_state == melee.Menu.IN_GAME and not done:
                
                if self._current_match_steps < self._max_match_steps:
                    for player in self.players:
                        if player.agent_type == agent_type.step_controlled_AI:
                            controller_actions = self._logical_actions_to_controller_actions_fn(logical_actions, i)
                            this_agent_controller = get_agent_controller(player)
                            execute_actions(this_agent_controller, controller_actions)
                        elif player.agent_type == agent_type.enemy_controlled_AI:
                            enemy_ai_raw_actions = player.trained_agent_act(self._gamestate_to_obs_space_fn(self.gamestate))
                            enemy_ai_logical_actions = MeleeEnv_v2.agent_actions_to_logical_actions_fn_v2(enemy_ai_raw_actions)
                            controller_actions = self._logical_actions_to_controller_actions_fn(enemy_ai_logical_actions, i)
                            this_agent_controller = get_agent_controller(player)
                            execute_actions(this_agent_controller, controller_actions)
                        else:
                            player.act(self.gamestate)

                
                else:
                    all_players_press_nothing(self.players)
                    return self._gamestate_to_obs_space_fn(self.gamestate), 0, True, True, infos


                self.gamestate = self.console.step()
                self._current_match_steps += 1

                done = self._is_done()

                rewards += self.calculate_rewards_v2(self._friendly_ports, self._enemy_ports, self.previous_gamestate, self.gamestate)
        
                self.previous_gamestate = self.gamestate

                if done:
                    all_players_press_nothing(self.players) # if A is pressed at the end, skips char select
                    break

        return self._gamestate_to_obs_space_fn(self.gamestate), rewards, done, truncated, infos

    def step_logical(self, logical_actions): # currently only supports 1v1. future: list of list for 2v2?
        done = self._is_done()
        rewards = 0
        truncated = None
        infos = {}

        #for player in self.players:
            #if player.character != self.get_characters()[player.controller.port]:
                #print('error! a player chose: ' + str(player.character) + ' but plays as ' + str(self.get_characters()[player.controller.port]))

        for i in range(0, self._action_repeat):
            if self.gamestate.menu_state == melee.Menu.IN_GAME and not done:
                
                if self._current_match_steps < self._max_match_steps:
                    for player in self.players:
                        if not player.agent_type == agent_type.step_controlled_AI:
                            player.act(self.gamestate)
                        else:
                            controller_actions = self._logical_actions_to_controller_actions_fn(logical_actions, i)
                            this_agent_controller = get_agent_controller(player)
                            execute_actions(this_agent_controller, controller_actions)
                
                else:
                    all_players_press_nothing(self.players)
                    return self._gamestate_to_obs_space_fn(self.gamestate), 0, True, True, infos


                self.gamestate = self.console.step()
                self._current_match_steps += 1

                done = self._is_done()

                rewards += self.calculate_rewards_v2(self._friendly_ports, self._enemy_ports, self.previous_gamestate, self.gamestate)
        
                self.previous_gamestate = self.gamestate

                if done:
                    all_players_press_nothing(self.players) # if A is pressed at the end, skips char select
                    break

        return self._gamestate_to_obs_space_fn(self.gamestate), rewards, done, truncated, infos

    @staticmethod
    def agent_actions_to_logical_actions_fn_v2(agent_actions):
        
        stick_dir = agent_actions[0]
        buttonA = agent_actions[1]
        buttonB = agent_actions[2]
        full_shield = agent_actions[3]

        logical_actions = []

        if stick_dir == 1:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_left_up)
        
        elif stick_dir == 3:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_up_right)

        elif stick_dir == 5:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_right_down)

        elif stick_dir == 7:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_down_left)

        elif stick_dir == 8:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_left)
        
        elif stick_dir == 4:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_right)

        elif stick_dir == 2:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_up)
        
        elif stick_dir == 6:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_down)

        if buttonA == 1:
            logical_actions.append(sam_utils.logical_inputs_v1.button_A)
        
        if buttonB == 1:
            logical_actions.append(sam_utils.logical_inputs_v1.button_B)
        
        if full_shield == 1:
            logical_actions.append(sam_utils.logical_inputs_v1.full_shield)
        
        #print('this frame agent action= ' + str(agent_actions) + ' | ' + str(logical_actions))

        return logical_actions

    def close(self):
        for player in self.players:
            if player.controller:
                player.controller.disconnect()
        self.gamestate = None
        self.previous_gamestate = None

        if self.console:
            self.console.stop()
        time.sleep(2)

    #old stuff
    def step_logical_old(self, logical_actions): # currently only supports 1v1. future: list of list for 2v2?
        done = self._is_done()
        rewards = None
        truncated = None
        infos = {}

        
        if self.gamestate.menu_state == melee.Menu.IN_GAME and not done:
            
            if self._current_match_steps < self._max_match_steps:
                for player in self.players:
                    if not player.agent_type == "AI":
                        player.act(self.gamestate)
                    else:
                        controller_actions = self._logical_actions_to_controller_actions_fn(logical_actions)
                        this_agent_controller = get_agent_controller(player)
                        execute_actions(this_agent_controller, controller_actions)
            
            else:
                all_players_press_nothing(self.players)
                return self._gamestate_to_obs_space_fn(self.gamestate), 0, True, True, infos


            self.gamestate = self.console.step()
            self._current_match_steps += 1

            done = self._is_done()

            rewards = self.calculate_rewards_v2(self._friendly_ports, self._enemy_ports, self.previous_gamestate, self.gamestate)
        
        self.previous_gamestate = self.gamestate
        

        if done:
            all_players_press_nothing(self.players) # if A is pressed at the end, skips char select

        return self._gamestate_to_obs_space_fn(self.gamestate), rewards, done, truncated, infos

def get_agent_controller(agent):
    return agent.controller

def execute_actions(controller, actions):
    controller.release_all()
    for action in actions:
        action(controller)
    controller.flush()

def _agent_actions_to_logical_actions_fn_v1(agent_actions):
        # old stuff with multibinary (was not working with dreamer stuff)
        #left = agent_actions[0] 
        #right = agent_actions[1]
        #up = agent_actions[2]
        #down = agent_actions[3]
        #buttonA = agent_actions[4]
        #buttonB = agent_actions[5]
        #full_shield = agent_actions[6]

        print('this frame agent action= ' + str(agent_actions))
        left = (agent_actions == 0)
        right = (agent_actions == 1)
        up = (agent_actions == 2)
        down = (agent_actions == 3)
        buttonA = (agent_actions == 4)
        buttonB = (agent_actions == 5)
        full_shield = (agent_actions == 6)

        logical_actions = []

        if (left and up) and not (right or down):
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_left_up)
        
        elif (up and right) and not (down or left):
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_up_right)

        elif (right and down) and not (left or up):
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_right_down)

        elif (down and left) and not (up or right):
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_down_left)

        elif left and not right:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_left)
        
        elif right and not left:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_right)

        elif up and not down:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_up)
        
        elif down and not up:
            logical_actions.append(sam_utils.logical_inputs_v1.joystick_down)

        if buttonA:
            logical_actions.append(sam_utils.logical_inputs_v1.button_A)
        
        if buttonB:
            logical_actions.append(sam_utils.logical_inputs_v1.button_B)
        
        if full_shield:
            logical_actions.append(sam_utils.logical_inputs_v1.full_shield)
        
        return logical_actions


def all_players_press_nothing(players):
    for player in players:
        player.controller.release_all()
        player.controller.flush()