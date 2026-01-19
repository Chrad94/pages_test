from ib_insync import IB, Stock, MarketOrder
import time
from ib_insync import IB, Stock
from tabulate import tabulate
import time
from typing import Dict, List, Optional
import yfinance as yf

def adjust_position_to_target(stock: str, target_usd: float):
    """
    Adjusts position in a stock to match target USD value.

    Args:
        stock: Stock ticker symbol (e.g., 'TSLA')
        target_usd: Target position value in USD

    Returns:
        Order object if trade executed, None otherwise
    """
    # Connect to IB
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    try:
        # Create contract
        contract = Stock(stock, 'SMART', 'USD')
        ib.qualifyContracts(contract)

        # Get current price using snapshot (doesn't require subscription)
        current_price = None

        # Method 1: Use reqTickers for snapshot data
        try:
            tickers = ib.reqTickers(contract)
            if tickers and len(tickers) > 0:
                ticker = tickers[0]

                # Try different price sources
                if ticker.marketPrice() == ticker.marketPrice():
                    current_price = ticker.marketPrice()
                    print(f"Using market price: ${current_price:.2f}")
                elif ticker.last == ticker.last:
                    current_price = ticker.last
                    print(f"Using last price: ${current_price:.2f}")
                elif ticker.close == ticker.close:
                    current_price = ticker.close
                    print(f"Using close price: ${current_price:.2f}")
        except Exception as e:
            print(f"reqTickers failed: {e}")

        # Method 2: Try delayed market data (free, no subscription needed)
        if current_price is None or current_price != current_price:
            print("Trying delayed market data...")
            ticker = ib.reqMktData(contract, '', False, False)

            # Request delayed data specifically (market data type 3 = delayed)
            ib.reqMarketDataType(3)  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
            ib.sleep(4)

            if ticker.last == ticker.last:
                current_price = ticker.last
                print(f"Using delayed last price: ${current_price:.2f}")
            elif ticker.close == ticker.close:
                current_price = ticker.close
                print(f"Using delayed close: ${current_price:.2f}")

            ib.cancelMktData(contract)

        # Method 3: Historical data as last resort
        if current_price is None or current_price != current_price:
            print("Fetching historical data...")
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='2 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1,
                    timeout=10
                )
                if bars and len(bars) > 0:
                    current_price = bars[-1].close
                    print(f"Using historical close: ${current_price:.2f}")
            except Exception as e:
                print(f"Historical data failed: {e}")

        if current_price is None or current_price != current_price:
            raise ValueError(f"Unable to get valid price for {stock}. Make sure only paper trading session is active.")

        print(f"\n✓ Current price of {stock}: ${current_price:.2f}")

        # Get current position
        positions = ib.positions()
        current_shares = 0
        current_value = 0

        for pos in positions:
            if pos.contract.symbol == stock:
                current_shares = pos.position
                current_value = current_shares * current_price
                break

        print(f"✓ Current position: {current_shares} shares (${current_value:.2f})")
        print(f"✓ Target position: ${target_usd:.2f}")

        # Calculate shares to trade
        target_shares = target_usd / current_price
        shares_to_trade = target_shares - current_shares

        # Round to whole shares
        shares_to_trade = round(shares_to_trade)

        if shares_to_trade == 0:
            print("\n✓ Position already at target. No trade needed.")
            return None

        # Determine action
        action = 'BUY' if shares_to_trade > 0 else 'SELL'
        quantity = abs(shares_to_trade)

        print(f"\n→ Placing order: {action} {quantity} shares of {stock}")

        # Create and place market order
        order = MarketOrder(action, quantity)
        trade = ib.placeOrder(contract, order)

        # Wait for order to fill
        ib.sleep(3)

        print(f"✓ Order status: {trade.orderStatus.status}")

        # Show filled price if available
        if trade.orderStatus.avgFillPrice:
            print(f"✓ Fill price: ${trade.orderStatus.avgFillPrice:.2f}")

        return trade

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

    finally:
        ib.disconnect()

class Portfolio:
    """
    Interactive Brokers Portfolio Manager
    """

    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Initialize Portfolio manager.

        Args:
            host: IB Gateway/TWS host
            port: Port (7497 for paper TWS, 7496 for live TWS)
            client_id: Unique client ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self._positions = []
        self._account_values = {}
        self._portfolio_data = []
        self._last_update = None

    def connect(self):
        """Connect to Interactive Brokers"""
        if self.ib is None or not self.ib.isConnected():
            self.ib = IB()
            self.ib.connect(self.host, self.port, self.client_id)
            self.ib.reqMarketDataType(3)  # Delayed data
            print("✓ Connected to Interactive Brokers")

    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            print("✓ Disconnected from Interactive Brokers")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def refresh(self):
        """
        Refresh portfolio data from IB.
        Fetches positions, account values, and current prices.
        """
        if not self.ib or not self.ib.isConnected():
            self.connect()

        print("\n🔄 Refreshing portfolio data...")

        # Get positions
        self._positions = self.ib.positions()

        # Get account summary - handle non-numeric values
        account_summary = self.ib.accountSummary()
        self._account_values = {}

        for item in account_summary:
            try:
                # Try to convert to float
                self._account_values[item.tag] = float(item.value)
            except (ValueError, TypeError):
                # Keep as string if not numeric
                self._account_values[item.tag] = item.value

        # Prepare portfolio data
        self._portfolio_data = []

        if not self._positions:
            print("✓ No positions found")
            self._last_update = time.time()
            return

        for pos in self._positions:
            # Skip non-stock positions
            if pos.contract.secType != 'STK':
                continue

            symbol = pos.contract.symbol
            shares = pos.position
            avg_cost = pos.avgCost

            # Get current price
            ticker = self.ib.reqMktData(pos.contract, '', False, False)
            self.ib.sleep(1.2)

            current_price = None
            if ticker.last == ticker.last:
                current_price = ticker.last
            elif ticker.close == ticker.close:
                current_price = ticker.close
            elif ticker.marketPrice() == ticker.marketPrice():
                current_price = ticker.marketPrice()

            if current_price is None or current_price != current_price:
                current_price = avg_cost

            market_value = shares * current_price
            cost_basis = shares * avg_cost
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0

            self._portfolio_data.append({
                'symbol': symbol,
                'shares': shares,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'contract': pos.contract
            })

            self.ib.cancelMktData(pos.contract)

        # Sort by market value
        self._portfolio_data.sort(key=lambda x: x['market_value'], reverse=True)

        self._last_update = time.time()
        print(f"✓ Portfolio refreshed ({len(self._portfolio_data)} positions)")

    def get_assets(self) -> List[str]:
        """
        Get list of stock symbols in portfolio.

        Returns:
            List of stock ticker symbols
        """
        if not self._portfolio_data:
            self.refresh()
        return [pos['symbol'] for pos in self._portfolio_data]

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position details for a specific stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with position details or None if not found
        """
        if not self._portfolio_data:
            self.refresh()

        for pos in self._portfolio_data:
            if pos['symbol'] == symbol:
                return pos.copy()
        return None

    def get_total_value(self) -> float:
        """
        Get total portfolio value in USD.

        Returns:
            Total market value of all positions
        """
        if not self._portfolio_data:
            self.refresh()

        return sum(pos['market_value'] for pos in self._portfolio_data)

    def get_buying_power(self) -> float:
        """
        Get available buying power (tradeable amount).

        Returns:
            Available funds for trading
        """
        if not self._account_values:
            self.refresh()

        # BuyingPower for margin accounts, AvailableFunds for cash accounts
        buying_power = self._account_values.get('BuyingPower',
                                                self._account_values.get('AvailableFunds', 0))

        # Ensure it's numeric
        try:
            return float(buying_power)
        except (ValueError, TypeError):
            return 0.0

    def get_cash_balance(self) -> float:
        """
        Get cash balance in account.

        Returns:
            Cash balance in USD
        """
        if not self._account_values:
            self.refresh()

        cash = self._account_values.get('TotalCashValue', 0)
        try:
            return float(cash)
        except (ValueError, TypeError):
            return 0.0

    def get_net_liquidation(self) -> float:
        """
        Get net liquidation value (total account value).

        Returns:
            Net liquidation value in USD
        """
        if not self._account_values:
            self.refresh()

        net_liq = self._account_values.get('NetLiquidation', 0)
        try:
            return float(net_liq)
        except (ValueError, TypeError):
            return 0.0

    def get_allocations(self) -> Dict[str, float]:
        """
        Get portfolio allocation percentages.

        Returns:
            Dictionary mapping symbols to allocation percentages
        """
        if not self._portfolio_data:
            self.refresh()

        total_value = self.get_total_value()
        if total_value == 0:
            return {}

        return {
            pos['symbol']: (pos['market_value'] / total_value * 100)
            for pos in self._portfolio_data
        }

    def get_total_pnl(self) -> Dict[str, float]:
        """
        Get total unrealized P&L.

        Returns:
            Dictionary with 'usd' and 'pct' keys
        """
        if not self._portfolio_data:
            self.refresh()

        total_pnl = sum(pos['unrealized_pnl'] for pos in self._portfolio_data)
        total_cost = sum(pos['cost_basis'] for pos in self._portfolio_data)
        pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0

        return {
            'usd': total_pnl,
            'pct': pnl_pct
        }

    def print_portfolio(self, show_pnl: bool = True):
        """
        Print formatted portfolio table.

        Args:
            show_pnl: Whether to show P&L columns
        """
        if not self._portfolio_data:
            self.refresh()

        if not self._portfolio_data:
            print("\n📊 PORTFOLIO: Empty")
            return

        # Prepare table data
        headers = ['Symbol', 'Shares', 'Avg Cost', 'Current', 'Value', 'Allocation']
        if show_pnl:
            headers.extend(['P&L ($)', 'P&L (%)'])

        table_data = []
        total_value = self.get_total_value()

        for pos in self._portfolio_data:
            allocation = (pos['market_value'] / total_value * 100) if total_value > 0 else 0

            row = [
                pos['symbol'],
                f"{pos['shares']:.2f}",
                f"${pos['avg_cost']:.2f}",
                f"${pos['current_price']:.2f}",
                f"${pos['market_value']:,.2f}",
                f"{allocation:.1f}%"
            ]

            if show_pnl:
                pnl_symbol = '🟢' if pos['unrealized_pnl'] >= 0 else '🔴'
                row.extend([
                    f"{pnl_symbol} ${pos['unrealized_pnl']:,.2f}",
                    f"{pos['unrealized_pnl_pct']:+.2f}%"
                ])

            table_data.append(row)

        # Print table
        print("\n" + "=" * 90)
        print("📊 PORTFOLIO SUMMARY")
        print("=" * 90)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))

        # Print totals
        total_pnl = self.get_total_pnl()
        net_liq = self.get_net_liquidation()
        cash = self.get_cash_balance()
        buying_power = self.get_buying_power()

        print("\n" + "=" * 90)
        print("💰 ACCOUNT SUMMARY")
        print("=" * 90)
        print(f"{'Net Liquidation:':<25} ${net_liq:>15,.2f}")
        print(f"{'Stock Positions:':<25} ${total_value:>15,.2f}")
        print(f"{'Cash Balance:':<25} ${cash:>15,.2f}")
        print(f"{'Buying Power:':<25} ${buying_power:>15,.2f}")

        if show_pnl:
            pnl_color = '🟢' if total_pnl['usd'] >= 0 else '🔴'
            print(f"{'Unrealized P&L:':<25} {pnl_color} ${total_pnl['usd']:>12,.2f} ({total_pnl['pct']:+.2f}%)")

        print("=" * 90 + "\n")

    def print_summary(self):
        """Print quick portfolio summary (no table)"""
        if not self._portfolio_data:
            self.refresh()

        total_value = self.get_total_value()
        net_liq = self.get_net_liquidation()
        buying_power = self.get_buying_power()
        pnl = self.get_total_pnl()

        print("\n💼 Quick Summary")
        print("-" * 50)
        print(f"Positions:        {len(self._portfolio_data)}")
        print(f"Total Value:      ${total_value:,.2f}")
        print(f"Net Liquidation:  ${net_liq:,.2f}")
        print(f"Buying Power:     ${buying_power:,.2f}")
        print(f"Unrealized P&L:   ${pnl['usd']:,.2f} ({pnl['pct']:+.2f}%)")
        print("-" * 50 + "\n")

    def get_concentration_risk(self, threshold: float = 20.0) -> List[str]:
        """
        Identify positions that exceed allocation threshold.

        Args:
            threshold: Percentage threshold (default 20%)

        Returns:
            List of symbols exceeding threshold
        """
        allocations = self.get_allocations()
        return [symbol for symbol, alloc in allocations.items() if alloc > threshold]

    def export_to_dict(self) -> Dict:
        """
        Export portfolio data as dictionary.

        Returns:
            Complete portfolio data
        """
        if not self._portfolio_data:
            self.refresh()

        return {
            'positions': self._portfolio_data.copy(),
            'total_value': self.get_total_value(),
            'net_liquidation': self.get_net_liquidation(),
            'cash_balance': self.get_cash_balance(),
            'buying_power': self.get_buying_power(),
            'total_pnl': self.get_total_pnl(),
            'allocations': self.get_allocations(),
            'last_update': self._last_update
        }

def get_sp500_index_value() -> Optional[float]:
    try:
        t = yf.Ticker("^GSPC")
        hist = t.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"⚠️ Failed to fetch S&P 500 index: {e}")
    return None

# Example usage:
if __name__ == "__main__":
    #adjust_position_to_target('FICO', 10000)
    #adjust_position_to_target('CAG', 10000)
    #adjust_position_to_target('GIS', 10000)
    #adjust_position_to_target('CSGP', 10000)
    #adjust_position_to_target('SBAC', 10000)

    print(get_sp500_index_value())

    # Method 1: Using context manager (recommended)
    with Portfolio() as portfolio:
        # Print full portfolio
        portfolio.print_portfolio()

        # Get specific information
        print(f"Assets: {portfolio.get_assets()}")
        print(f"Total Value: ${portfolio.get_total_value():,.2f}")
        print(f"Buying Power: ${portfolio.get_buying_power():,.2f}")

        # Check specific position
        aapl = portfolio.get_position('AAPL')
        if aapl:
            print(f"\nAAPL Position: {aapl['shares']} shares @ ${aapl['current_price']:.2f}")

        # Check concentration risk
        concentrated = portfolio.get_concentration_risk(threshold=15.0)
        if concentrated:
            print(f"\n⚠️  Concentrated positions (>15%): {concentrated}")

    # Method 2: Manual connection management
    """
    portfolio = Portfolio()
    portfolio.connect()

    portfolio.print_summary()
    portfolio.print_portfolio()

    portfolio.disconnect()
    """