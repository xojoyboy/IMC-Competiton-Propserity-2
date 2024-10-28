from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle


class Trader:

    POSITION_LIMIT = {
        'STARFRUIT': 100,  # Example limit, adjust as needed
        'AMETHYSTS': 50,   # Example limit, adjust as needed
    }

    def calculate_new_cost_price(self, current_cost_price, current_position, new_trade_price, new_trade_quantity):
        if current_position + new_trade_quantity == 0:
            return 0
        else:
            total_cost = (current_cost_price * current_position) + \
                (new_trade_price * new_trade_quantity)
            new_position = current_position + new_trade_quantity
            new_cost_price = total_cost / new_position
            return new_cost_price

    def calculate_realized_pnl(self, trade, cost_price):
        """
        Calculate realized PnL based on trade price, quantity, and cost price.
        """
        if trade.buyer == "SUBMISSION":
            # For buy orders, realized PnL is not directly affected.
            return 0
        elif trade.seller == "SUBMISSION":
            # For sell orders, realized PnL is the difference between trade price and cost price, times quantity.
            return (trade.price - cost_price) * trade.quantity

    def calculate_unrealized_pnl(self, current_position, market_price, cost_price):
        """
        Calculate unrealized PnL based on current position, market price, and cost price.
        """
        return (market_price - cost_price) * current_position

    def run(self, state: TradingState):
        if state.traderData:
            trader_state = jsonpickle.decode(state.traderData)
        else:
            trader_state = {'cost_price': {}, 'realized_pnl': 0, 'unrealized_pnl': 0}

        for product in state.listings.keys():
            if product not in trader_state['cost_price']:
                trader_state['cost_price'][product] = 0

        # Process trades to update cost prices and calculate P&L
        for product, trades in state.own_trades.items():
            for trade in trades:
                # Update cost price for buy transactions
                if trade.buyer == "SUBMISSION":
                    trader_state['cost_price'][product] = self.calculate_new_cost_price(
                        trader_state['cost_price'].get(product, 0),
                        state.position.get(product, 0) - trade.quantity,
                        trade.price,
                        trade.quantity
                    )
                # Calculate realized PnL
                trader_state['realized_pnl'] += self.calculate_realized_pnl(trade, trader_state['cost_price'][product])

        # Calculate unrealized PnL for each product based on the current market price and cost price
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            market_price = (best_ask + best_bid) / 2 if best_ask != float('inf') and best_bid != 0 else 0
            current_position = state.position.get(product, 0)
            
            trader_state['unrealized_pnl'] += self.calculate_unrealized_pnl(current_position, market_price, trader_state['cost_price'].get(product, 0))

        # Serialize and pass the updated trader state
        traderData = jsonpickle.encode(trader_state)

        print(f"Realized PnL: {trader_state['realized_pnl']}\n Unrealized PnL: {trader_state['unrealized_pnl']}")

        ############################################################################################################
        # Trading logic goes here
        result = {}  
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
        ############################################################################################################
        
        conversions = 0  # Assuming no conversions for simplicity

        return result, conversions, traderData