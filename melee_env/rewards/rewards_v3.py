def _get_stocks(self, gamestate):
    return {port : int(gamestate.players[port].stock) for port in gamestate.players.keys()}

def _get_damages(self, gamestate):
    return {port: int(gamestate.players[port].percent) for port in gamestate.players.keys()}


def calculate_rewards_v3(friendly_ports, enemy_ports,  previous_gamestate, current_gamestate):
    
    stock_multiplier = 1000

    if not previous_gamestate:
        return 0

    previous_stocks = _get_stocks(previous_gamestate)
    current_stocks = _get_stocks(current_gamestate)

    stock_differential = {port: current_stocks[port] - previous_stocks[port] for port in previous_stocks}

    stock_rewards = (sum(stock_differential[port] for port in friendly_ports) \
                    - sum(stock_differential[port] for port in enemy_ports)) \
                    * stock_multiplier

    rewards = stock_rewards 
    return rewards