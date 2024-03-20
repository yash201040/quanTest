class Quantester:
    def __init__(self, bt) -> None:
        self.bt = bt
        self.static_trades_df = pd.DataFrame()
        self.compounded_trades_df = pd.DataFrame()

    def calculate_trade_metrics(self):
        # Initialize variables to track trade metrics
        entry_price = 0
        position = 0
        trade = {}

        for _, row in self.bt.iterrows():
            if row["position"] != 0 and position == 0:  # Entry point
                entry_price = row["close"]
                position = row["position"]
                trade = {
                    "entry": entry_price,
                    "crest": entry_price,
                    "trough": entry_price,
                }

            elif row["position"] == 0 and position != 0:  # Exit point
                trade["exit"] = row["close"]
                trade["pnl"] = self.bt.calculate_trade_pnl(
                    trade["entry"], trade["exit"], position
                )
                trade["position"] = position
                # Additional metrics calculations here

                # Append to respective DataFrames
                self.static_trades_df = self.static_trades_df.append(
                    trade, ignore_index=True
                )
                # Reset for next trade
                position = 0

            # Update crest and trough
            if position != 0:
                if position == 1:
                    trade["crest"] = max(trade["crest"], row["high"])
                elif position == -1:
                    trade["trough"] = min(trade["trough"], row["low"])

                # m2m_pnl calculation and update balance columns in self.df
                row["m2m_pnl"] = position * (row["close"] - entry_price)

        # Finalize balances and other metrics for each row in self.df here
