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


class MeleeEnv_v2:
    def __init__(self, 
        iso_path,
        players,
        agent_actions_to_logical_actions_fn,
        logical_actions_to_controller_actions_fn,
        gamestate_to_obs_space_fn,
        fast_forward=False, 
        blocking_input=True,
        ai_starts_game=True,
        shuffle_controllers_after_each_game=True,
        randomize_stage = True,
        randomize_character = True):

        self.d = DolphinConfig()
        self.d.set_ff(fast_forward)

        self.iso_path = Path(iso_path).resolve()
        self.players = players

        self.step = self._gen_step(agent_actions_to_logical_actions_fn)
        self._logical_actions_to_controller_actions_fn = logical_actions_to_controller_actions_fn
        self._gamestate_to_obs_space_fn = gamestate_to_obs_space_fn

        # inform other players of other players
        # for player in self.players:
        #     player.set_player_keys(len(self.players))
        
        if len(self.players) == 2:
            self.d.set_center_p2_hud(True)
        else:
            self.d.set_center_p2_hud(False)

        self.blocking_input = blocking_input
        self.ai_starts_game = ai_starts_game

        self._shuffle_controllers_after_each_game = shuffle_controllers_after_each_game
        self._randomize_stage = randomize_stage
        self._randomize_character = randomize_character

        self.gamestate = None
        self.previous_gamestate = None
        self.env_is_started = False

        self._friendly_ports = None
        self._enemy_ports = None

        self._dead_ports = {}
        self.removethis = 0


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
            gfx_backend='Vulkan')

        # print(self.console.dolphin_home_path)  # add to logging later
        # Configure Dolphin for the correct controller setup, add controllers
        human_detected = False

        for i in range(len(self.players)):
            curr_player = self.players[i]
            if curr_player.agent_type == "HMN":
                self.d.set_controller_type(i+1, enums.ControllerType.GCN_ADAPTER)
                curr_player.controller = melee.Controller(console=self.console, port=i+1, type=melee.ControllerType.GCN_ADAPTER)
                #curr_player.port = i+1
                human_detected = True
            elif curr_player.agent_type in ["AI", "CPU", "HardCoded"]:
                self.d.set_controller_type(i+1, enums.ControllerType.STANDARD)
                curr_player.controller = melee.Controller(console=self.console, port=i+1)
                self.menu_control_agent = curr_player
                #curr_player.port = i+1 
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
 
    # so the agent learns to play each port
    def _shuffle_controllers(self):
        remaining_controllers = [player.controller for player in self.players]

        for player in self.players:
            chosen_controller = random.choice(remaining_controllers)
            player.controller = chosen_controller
            remaining_controllers.remove(chosen_controller)

    def _choose_stage(self):
        if self._randomize_stage:
            stage =  random.choice([
                #melee.enums.Stage.FINAL_DESTINATION,
                #melee.enums.Stage.BATTLEFIELD,
                melee.enums.Stage.POKEMON_STADIUM,])
                #melee.enums.Stage.DREAMLAND,
                #melee.enums.Stage.FOUNTAIN_OF_DREAMS,
                #melee.enums.Stage.YOSHIS_STORY])
        else:
            stage = melee.enums.Stage.BATTLEFIELD
        print('chosen stage = ' + str(stage))
        return stage

    def _randomize_characters(self):
        for player in self.players:
            if type(player) is NOOP:
                player.character = random.choice([
                    #melee.enums.Character.BOWSER,
                    #melee.enums.Character.CPTFALCON,
                    #melee.enums.Character.DK,
                    #melee.enums.Character.DOC,
                    #melee.enums.Character.FALCO,
                    #melee.enums.Character.FOX,
                    #melee.enums.Character.GAMEANDWATCH,
                    #melee.enums.Character.GANONDORF,
                    melee.enums.Character.JIGGLYPUFF,
                    #melee.enums.Character.KIRBY,
                    #melee.enums.Character.LINK,
                    #melee.enums.Character.LUIGI,
                    #melee.enums.Character.MARIO,
                    #melee.enums.Character.MARTH,
                    #melee.enums.Character.MEWTWO,
                    #melee.enums.Character.NESS,
                    #melee.enums.Character.PEACH,
                    #melee.enums.Character.PICHU,
                    melee.enums.Character.PIKACHU,])
                    #melee.enums.Character.POPO,
                    #melee.enums.Character.ROY,
                    #melee.enums.Character.SAMUS,
                    #melee.enums.Character.SHEIK, it makes it bug
                    #melee.enums.Character.YLINK,
                    #melee.enums.Character.YOSHI,
                    #melee.enums.Character.ZELDA])
                print('chosen char=' + str(player.character))

    def _populate_friendly_enemy_ports(self):
        self._friendly_ports = []
        self._enemy_ports = []
        
        for player in self.players:
            if player.agent_type == "AI": #todo: replace all othose with an enum
                self._friendly_ports.append(player.controller.port)
            else:
                self._enemy_ports.append(player.controller.port)

        

    def setup(self):
        self.previous_gamestate = None

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
            self.gamestate = self.console.step()

            if self.gamestate.menu_state is melee.Menu.CHARACTER_SELECT:
                for i in range(len(self.players)):
                    if self.players[i].agent_type in ["AI", "HardCoded"] :
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            start= (i==len(self.players) - 1))
                            #start=self.players[i].press_start)
                    if self.players[i].agent_type == "CPU":
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            cpu_level=self.players[i].lvl,
                            start= (i==len(self.players) - 1))
                            #start=self.players[i].press_start)  

            elif self.gamestate.menu_state is melee.Menu.STAGE_SELECT:
                print('choosin stage with controller port: ' + str(self.menu_control_agent.controller.port))
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
    
    # returns rewards only for the agent being trained
    def calculate_rewards_v2(self, friendly_ports, enemy_ports, previous_gamestate, current_gamestate):
        # todo: take as arg
        stock_multiplier = 200
        damage_multiplier = 1
        win_multiplier = 0

        if not previous_gamestate:
            return 0

        previous_stocks = self.get_stocks_v2(previous_gamestate)
        current_stocks = self.get_stocks_v2(current_gamestate)

        stock_differential = {port: current_stocks[port] - previous_stocks[port] for port in current_stocks}

        #if done and (stock_differential[port] == 0 for port in stock_differential):
            #return 0 

        for port in [port for port in stock_differential if stock_differential[port] != 0]:
            self._dead_ports[port] = True

        previous_damages = self.get_damages_v2(previous_gamestate)
        current_damages = self.get_damages_v2(current_gamestate)

        damages_differential = {}
        for port in current_damages:
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
    
    def reset(self):
        if not self.env_is_started:
            self.start()
        
        obs, done = self.setup()
        return self._gamestate_to_obs_space_fn(obs), None # obs, info

    # todo: fix for 4 players and make it cleaner
    def _is_done(self):
        stocks = np.array([self.gamestate.players[i].stock for i in list(self.gamestate.players.keys())])
        return not np.sum(stocks[np.argsort(stocks)][::-1][1:])
    
    def _gen_step(self, agent_to_logical_actions_fn):
        def step(agent_actions):
            logical_actions = agent_to_logical_actions_fn(agent_actions)
            return self.step_logical(logical_actions)
        return step

    def step_logical(self, logical_actions): # currently only supports 1v1. future: list of list for 2v2?
        done = self._is_done()
        rewards = None
        truncated = None
        infos = None

        if self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH] and not done:
            
            for player in self.players:
                if not player.agent_type == "AI":
                    player.act(self.gamestate)
                else:
                    controller_actions = self._logical_actions_to_controller_actions_fn(logical_actions)
                    this_agent_controller = get_agent_controller(player)
                    execute_actions(this_agent_controller, controller_actions)

            self.gamestate = self.console.step()
            #done = self._is_done()
            rewards = self.calculate_rewards_v2(self._friendly_ports, self._enemy_ports, self.previous_gamestate, self.gamestate)
            self.previous_gamestate = self.gamestate

        return self._gamestate_to_obs_space_fn(self.gamestate), rewards, done, truncated, infos


    def close(self):
        for t, c in self.controllers.items():
            c.disconnect()
        self.observation_space._reset()
        self.gamestate = None
        self.console.stop()
        time.sleep(2)

def get_agent_controller(agent):
    return agent.controller

def execute_actions(controller, actions):
    for action in actions:
        action(controller)