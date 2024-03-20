class Backtester:
    def __init__(self, df, strategy, initial_balance=1000):
        self.df = df.copy()  # columns: datetime, open, high, low, close, ...any
        self.strategy = strategy  # callable strat function returning signals list
        self.strategy_name = strategy.__name__  # strat function name
        self.initial_balance = initial_balance  # unrestricted to asset price
        self.df["signal"] = 0  # ± qty to trade
        self.df["position"] = 0  # net active position size
        self.df["pnl"] = 0.0  # mark-to-mark p&l of active trades
        self.df["arithmetic_balance"] = initial_balance  # with constant capital
        self.df["compounded_balance"] = initial_balance  # with changing capital

    def generate_signals(self):
        self.df["signal"] = self.strategy(self.df)
        self.df["position"] = self.df["signal"].cumsum()

    def calculate_pnl(self, entry_price, close_price, position):
        return (
            position
            * (close_price - entry_price)
            * (self.initial_balance / entry_price)
        )

    def run_backtest(self):
        self.generate_signals()
        # Additional calculations to finalize self.stat_df and self.comp_df
