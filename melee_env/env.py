from melee_env.dconfig import DolphinConfig
import melee
from melee import enums
import numpy as np
import sys
import time
from pathlib import Path
import copy


class MeleeEnv:
    def __init__(self, 
        iso_path,
        players,
        fast_forward=False, 
        blocking_input=True,
        ai_starts_game=True):

        self.d = DolphinConfig()
        self.d.set_ff(fast_forward)

        self.iso_path = Path(iso_path).resolve()
        self.players = players

        # inform other players of other players
        # for player in self.players:
        #     player.set_player_keys(len(self.players))
        
        if len(self.players) == 2:
            self.d.set_center_p2_hud(True)
        else:
            self.d.set_center_p2_hud(False)

        self.blocking_input = blocking_input
        self.ai_starts_game = ai_starts_game

        self.gamestate = None
        self.previous_gamestate = None
        self.env_is_started = False


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
                curr_player.port = i+1
                human_detected = True
            elif curr_player.agent_type in ["AI", "CPU"]:
                self.d.set_controller_type(i+1, enums.ControllerType.STANDARD)
                curr_player.controller = melee.Controller(console=self.console, port=i+1)
                self.menu_control_agent = i
                curr_player.port = i+1 
            else:  # no player
                self.d.set_controller_type(i+1, enums.ControllerType.UNPLUGGED)
            
        if self.ai_starts_game and not human_detected:
            self.ai_press_start = True

        else:
            self.ai_press_start = False  # don't let ai press start without the human player joining in. 

        if self.ai_starts_game and self.ai_press_start:
            self.players[self.menu_control_agent].press_start = True

        self.console.run(iso_path=self.iso_path)
        self.console.connect()

        [player.controller.connect() for player in self.players if player is not None]

        self.gamestate = self.console.step()
        self.env_is_started = True
 
    def setup(self, stage):
        self.previous_gamestate = None

        for player in self.players:
            player.defeated = False
            
        while True:
            self.gamestate = self.console.step()
            if self.gamestate.menu_state is melee.Menu.CHARACTER_SELECT:
                for i in range(len(self.players)):
                    if self.players[i].agent_type == "AI":
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            start=self.players[i].press_start)
                    if self.players[i].agent_type == "CPU":
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            cpu_level=self.players[i].lvl,
                            start=self.players[i].press_start)  

            elif self.gamestate.menu_state is melee.Menu.STAGE_SELECT:
                melee.MenuHelper.choose_stage(
                    stage=stage,
                    gamestate=self.gamestate,
                    controller=self.players[self.menu_control_agent].controller)

            elif self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return self.gamestate, False  # game is not done on start
                
            else:
                melee.MenuHelper.choose_versus_mode(self.gamestate, self.players[self.menu_control_agent].controller)
    
    def get_stocks(self, gamestate):
        stocks = [gamestate.players[i].stock for i in list(gamestate.players.keys())]
        return np.array([stocks]).T  # players x 1
  
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

    def calculate_rewards(self, previous_gamestate, current_gamestate):

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
    
    def reset(self):
        if not self.env_is_started:
            self.start()
        
        obs, done = self.setup(enums.Stage.BATTLEFIELD)
        return obs, None # obs, info

    def step(self):
        stocks = np.array([self.gamestate.players[i].stock for i in list(self.gamestate.players.keys())])
        done = not np.sum(stocks[np.argsort(stocks)][::-1][1:])
        rewards = None
        truncated = None
        infos = None

        if self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH] and not done:
            self.gamestate = self.console.step()
            rewards = self.calculate_rewards(self.previous_gamestate, self.gamestate)
            self.previous_gamestate = self.gamestate

        return self.gamestate, rewards, done, truncated, infos


    def close(self):
        for t, c in self.controllers.items():
            c.disconnect()
        self.observation_space._reset()
        self.gamestate = None
        self.console.stop()
        time.sleep(2)