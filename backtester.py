import numpy as np
import pandas as pd


class Backtester:
    """
    A class to backtest trading strategies on historical asset data.

    Attributes:
        df (pd.DataFrame): Asset dataframe containing columns datetime, open, high, low, and close.
        strategy (callable): A function that returns a list of trading signals based on the asset dataframe.
        strategy_name (str): The name of the trading strategy function.
        initial_balance (float): The starting balance for backtesting, in the same currency as the asset prices.
        tz_array (np.ndarray): A structured NumPy array representing the trading zones extracted from the asset data.
        trade_template (dict): A template for recording trade details during the backtest.
        tb_df (pd.DataFrame): A trade book dataframe, logging each trade executed during the backtest.
        long_tb_df (pd.DataFrame or None): The trade book data frame filtered for long positions.
        short_tb_df (pd.DataFrame or None): The trade book data frame filtered for short positions.
        sl_df (pd.DataFrame or None): Ledger dataframe computed with fixed deployed amounts for each trade.
        long_sl_df (pd.DataFrame or None): The static ledger data frame filtered for long positions.
        short_sl_df (pd.DataFrame or None): The static ledger data frame filtered for short positions.
        cl_df (pd.DataFrame or None): Ledger dataframe computed with compounding deployed amounts for each trade.
        long_cl_df (pd.DataFrame or None): The compounding ledger data frame filtered for long positions.
        short_cl_df (pd.DataFrame or None): The compounding ledger data frame filtered for short positions.
        summary_df (pd.DataFrame or None): Summary dataframe containing key metrics and statistics of the backtest.

    Methods:
        __init__(self, asset_df, strategy, initial_balance=10000): Initializes the Backtester class with the asset data,
            trading strategy, and an initial balance. Sets up the trade book, trade ledgers, and computes initial trade metrics.
    """

    def __init__(self, asset_df, strategy, initial_balance=10000):
        """
        Initializes the Backtester object with the given asset data, strategy, and initial balance.
        It preprocesses the data, sets up the necessary structures for backtesting, performs the backtest,
        and computes initial trading metrics.

        Parameters:
            asset_df (pd.DataFrame): The historical asset data to backtest on,
                                     with columns datetime, open, high, low, and close.
            strategy (callable): The trading strategy function to be backtested. It should accept the asset dataframe
                                 and return a list of trading signals.
            initial_balance (float, optional): The starting balance for the backtest.

        Raises:
            ValueError: If the input parameters are invalid or if the strategy function does not return the expected output.
        """
        # Assign and verify inputs
        self.df = asset_df  # cols - datetime, open, high, low, close
        self.strategy = (
            strategy  # callable returns list of trade signals without overlap
        )
        self.strategy_name = (
            strategy.__name__
        )  # Ex: mean_reversion_strategy, momentum_trading_strategy
        self.initial_balance = initial_balance  # same currency as asset price
        self._check_inputs()

        # Filter trading zones, create trade template, initialize trade book
        self.tz_array = self._generate_trading_zones()
        self.trade_template = {
            "start_time": 0,
            "end_time": 0,
            "position": 0,
            "entry": 0,
            "crest": 0,
            "trough": 0,
            "end": 0,
            "win": 0,
            "change_p": 0,
            "spike_p": 0,
            "dip_p": 0,
            "duration": 0,
        }
        self.tb_df = pd.DataFrame(columns=self.trade_template.keys())
        self.long_tb_df = None  # long trades
        self.short_tb_df = None  # short trades

        # Backtrade the strategy and save records in trade book
        self._back_trade()
        # Compute trade metrics
        self._compute_trade_metrics()

        # Initialize static trade ledgers
        self.sl_df, self.long_sl_df, self.short_sl_df = (
            None,
            None,
            None,
        )
        # Initialize compounding trade ledgers
        self.cl_df, self.long_cl_df, self.short_cl_df = (
            None,
            None,
            None,
        )
        # Initialize final backtest summary
        self.summary_df = None

    def _check_inputs(self):
        """
        Validates the inputs provided to the Backtester.

        This method checks that the asset data frame contains the required columns,
        the strategy is a callable function, and the initial balance is a positive number.

        Raises:
            ValueError: If missing columns in asset df or if initial balance is not +ve.
            KeyError: If required columns are missing in the asset data frame.
            TypeError: If the strategy provided is not a function.
        """
        df_message = (
            "Input a pandas data frame with columns - datetime, open, high, low, close."
        )
        strategy_message = (
            "Pass a strategy function reference that returns signals (0, ±1) in list."
        )
        balance_message = "Initial balance should be greater than zero."
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError(df_message)
        expected_cols = ["datetime", "open", "high", "low", "close"]
        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        if missing_cols:
            raise KeyError(df_message)
        if not callable(self.strategy):
            raise TypeError(strategy_message)
        if self.initial_balance <= 0:
            raise ValueError(balance_message)

    def _generate_trading_zones(self):
        """
        Generates a structured NumPy array containing the trading zones extracted from the asset data frame.

        This method processes the asset data to identify trading zones, where trade signals and positions are non-zero,
        and converts this filtered data into a structured NumPy array for efficient processing.

        Returns:
            np.ndarray: A structured NumPy array with fields for datetime, high, low, close, signal, and position,
                        representing the trading zones.
        """
        # Calculate all trade signals and positional states
        all_signals = np.array(self.strategy(self.df))
        all_positions = np.cumsum(all_signals)

        # Filter out rows where both signal and position are zero initially
        tz_indices = (all_signals != 0) | (np.roll(all_positions, -1) != 0)
        tz_signals = all_signals[tz_indices]
        tz_positions = all_positions[tz_indices]
        # Create a trading zone data frame
        tz_df = self.df.loc[tz_indices, ["datetime", "high", "low", "close"]]

        # Check and remove the last unclosed trading zone if exists
        if tz_positions[-1] != 0:
            last_trade_end_index = np.max(np.nonzero(tz_signals)[0])
            tz_signals = tz_signals[: last_trade_end_index + 1]
            tz_positions = tz_positions[: last_trade_end_index + 1]
            tz_df = tz_df.iloc[: last_trade_end_index + 1]

        # Define a structured dtype for the NumPy array
        dtype = [
            ("datetime", "datetime64[ns]"),
            ("high", "float64"),
            ("low", "float64"),
            ("close", "float64"),
            ("signal", "int32"),
            ("position", "int32"),
        ]

        # Create a structured numpy array
        tz_array = np.empty(tz_df.shape[0], dtype=dtype)
        tz_array["datetime"] = tz_df["datetime"].values
        tz_array["high"] = tz_df["high"].values
        tz_array["low"] = tz_df["low"].values
        tz_array["close"] = tz_df["close"].values
        tz_array["signal"] = tz_signals
        tz_array["position"] = tz_positions

        return tz_array

    def _back_trade(self):
        """
        Executes the backtesting process using the trading strategy and asset data.

        This method iterates through the trading zones, applies the trading strategy to make trades,
        updates the trade book, and captures trade details such as entry, exit, highs, and lows.

        After completing the trades, it appends the trade data to the trade book data frame (self.tb_df).
        """
        # Create a list to store all trades
        trades = []
        trade = dict(self.trade_template)
        active_position = 0

        # Iterate in trading zones using a NumPy structured array
        for row in self.tz_array:
            if row["signal"] != 0:  # discover order point
                if active_position == 0:  # no running position
                    # Enter fresh trade
                    trade = dict(self.trade_template)
                    self._enter_trade(trade, row)
                else:  # a position is already running
                    # Update swings, close position and save trade
                    self._update_swings(trade, row, active_position)
                    self._end_trade(trade, row)
                    trades.append(trade)
                    if np.sign(active_position) != np.sign(row["position"]):
                        # Flip previous position to take opp. trade
                        trade = dict(self.trade_template)
                        self._enter_trade(trade, row)
                active_position = row["position"]
            else:  # ongoing trade
                self._update_swings(trade, row, active_position)

        # Add all trades data to the trade book data frame
        self.tb_df = pd.concat([self.tb_df, pd.DataFrame(trades)], ignore_index=True)

    def _enter_trade(self, trade, row):
        """
        Initializes the trade dictionary with values for a new trade entry.

        Parameters:
            trade (dict): The trade record being populated.
            row (namedtuple or pd.Series): The current row of trading data being processed.
        """
        trade["position"] = row["position"]
        trade["start_time"] = row["datetime"]
        trade["entry"] = row["close"]
        trade["crest"] = row["close"]
        trade["trough"] = row["close"]

    def _end_trade(self, trade, row):
        """
        Finalizes the trade dictionary with values to close the trade.

        Parameters:
            trade (dict): The trade record being updated.
            row (namedtuple or pd.Series): The current row of trading data being processed.
        """
        trade["end_time"] = row["datetime"]
        trade["end"] = row["close"]

    def _update_swings(self, trade, row, active_position):
        """
        Updates the crest and trough values in the trade dictionary based on new price data.

        Parameters:
            trade (dict): The trade record being updated.
            row (namedtuple or pd.Series): The current row of trading data being processed.
            active_position (int): The current active position, indicating the trade's direction.
        """
        if active_position > 0:  # long trade
            trade["crest"] = max(trade["crest"], row["high"])
            trade["trough"] = min(trade["trough"], row["low"])
        else:  # short trade
            trade["crest"] = min(trade["crest"], row["low"])
            trade["trough"] = max(trade["trough"], row["high"])

    def _compute_trade_metrics(self):
        """
        Computes and assigns trade metrics such as win rate, percentage changes, and durations to the trade book dataframe.
        """
        # Assign tb_df for quick access
        tb_df = self.tb_df

        # Ensure data types are correct, especially for datetime operations
        tb_df["start_time"] = pd.to_datetime(tb_df["start_time"])
        tb_df["end_time"] = pd.to_datetime(tb_df["end_time"])

        # Long trades win when end > entry, short trades win when end < entry
        tb_df["win"] = ((tb_df["end"] > tb_df["entry"]) & (tb_df["position"] > 0)) | (
            (tb_df["end"] < tb_df["entry"]) & (tb_df["position"] < 0)
        )
        tb_df["win"] = tb_df["win"].astype(int)

        # Calculate the percentage change, spike, and dip for each trade
        tb_df["change_p"] = (
            (tb_df["end"] / tb_df["entry"] - 1) * 100 * np.sign(tb_df["position"])
        )
        tb_df["spike_p"] = (
            (tb_df["crest"] / tb_df["entry"] - 1) * 100 * np.sign(tb_df["position"])
        )
        tb_df["dip_p"] = (
            (tb_df["trough"] / tb_df["entry"] - 1) * 100 * np.sign(tb_df["position"])
        )

        # Compute the duration of each trade
        tb_df["duration"] = tb_df["end_time"] - tb_df["start_time"]

    def generate_ledger(self, mode="static"):
        """
        Generates the trading ledger based on the specified mode (static or compounding).

        Parameters:
            mode (str): The mode of trading ledger to generate. Options are 'static' or 'compounding'.

        Returns:
            pd.DataFrame: The generated ledger dataframe based on the specified mode.

        Raises:
            ValueError: If an invalid mode is specified.
        """
        if mode == "static":
            if self.sl_df is None:
                self._generate_static_ledger()
            return self.sl_df
        elif mode == "compounding":
            if self.cl_df is None:
                self._generate_compounding_ledger()
            return self.cl_df
        else:
            raise ValueError("Invalid mode. Use 'static' or 'compounding'.")

    def _generate_static_ledger(self):
        """
        Generates the static ledger dataframe using a fixed deployed amount for each trade.
        """
        # Create static ledger df
        sl_df = self.tb_df[["position", "change_p"]].copy()

        # Initialize starting amounts
        current_balance = self.initial_balance
        deployed = self.initial_balance
        balances = []
        pnls = []

        for row in sl_df.itertuples(index=False):
            pnl = (row.change_p / 100) * deployed
            current_balance += pnl  # Update balance after each trade
            balances.append(current_balance)  # Reflects balance after the trade
            pnls.append(pnl)

        # Set the deployed and balance columns
        sl_df["deployed"] = [deployed] * len(sl_df)
        sl_df["balance"] = balances
        sl_df["pnl"] = pnls

        self.sl_df = sl_df

    def _generate_compounding_ledger(self):
        """
        Generates the compounding ledger dataframe where each trade's deployed amount is the entire current balance.
        """
        # Create compounding ledger df
        cl_df = self.tb_df[["position", "change_p"]].copy()

        # Initialize starting amounts
        current_balance = self.initial_balance
        deployed_amounts = []
        balances = []
        pnls = []

        for row in cl_df.itertuples(index=False):
            deployed = current_balance
            pnl = (row.change_p / 100) * deployed
            current_balance += pnl
            deployed_amounts.append(deployed)
            balances.append(current_balance)  # Reflects balance after the trade
            pnls.append(pnl)

            # Check if broke, stop trading
            if current_balance <= 0:
                break

        remaining_trade_count = len(deployed_amounts) - len(cl_df)
        if remaining_trade_count < 0:
            deployed_amounts.extend([0] * remaining_trade_count)
            balances.extend([balances[-1]] * remaining_trade_count)
            pnls.extend([0] * remaining_trade_count)

        # Add deployed and balance amounts to data frame
        cl_df["deployed"] = deployed_amounts
        cl_df["balance"] = balances
        cl_df["pnl"] = pnls

        self.cl_df = cl_df

    def generate_summary_df(self):
        """
        Generates a summary dataframe containing key metrics and statistics of the backtesting results.

        The summary includes metrics such as the number of trades, accuracy, growth rates, drawdowns, and various
        performance indicators for long, short, and all trades.

        Returns:
            pd.DataFrame: A dataframe with the summary of backtesting results.
        """
        if self.summary_df:
            return self.summary_df

        # THIS METHOD IS WORK IN PROGRESS

    def _separate_long_short_trades(self):
        """
        Separates the trades into long and short categories and integrates ledger data if available.

        This method divides the trades based on their position (long or short) and also separates the static and compounding
        ledger dataframes into long and short trades if they exist.

        Returns:
            tuple: A tuple containing dataframes for long trades, short trades, long and short static ledgers,
                   and long and short compounding ledgers.
        """
        # Separate trade book into long and short type positions
        self.long_tb_df = self.tb_df[self.tb_df["position"] > 0]
        self.short_tb_df = self.tb_df[self.tb_df["position"] < 0]

        # Slice ledger data if available into long and short type positions
        if self.sl_df is not None:
            self.long_sl_df = self.sl_df[self.sl_df["position"] > 0]
            self.short_sl_df = self.sl_df[self.sl_df["position"] < 0]
        if self.cl_df is not None:
            self.long_cl_df = self.cl_df[self.cl_df["position"] > 0]
            self.short_cl_df = self.cl_df[self.cl_df["position"] < 0]

    def _calculate_drawdown(self, ledger_df):
        """
        Calculates the maximum drawdown percentage in the ledger.

        The drawdown is the maximum observed loss from a peak to a trough in the balance over a period,
        before a new peak is achieved.

        Parameters:
            ledger_df (pd.DataFrame): The ledger dataframe containing balance information.

        Returns:
            float: The maximum drawdown percentage observed in the ledger.
        """
        max_drawdown = 0
        peak_balance = ledger_df["balance"].iloc[0]

        for balance in ledger_df["balance"]:
            if balance > peak_balance:
                peak_balance = balance  # New peak found
            else:
                drawdown = 1 - balance / peak_balance
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return max_drawdown * 100  # Convert to percentage
