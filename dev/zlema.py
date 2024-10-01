import pandas as pd

def zlema(
        X: pd.Series | pd.DataFrame,
        lag: int,
        K=0.5,
        ewm_kwargs: dict = {"adjust": False},
    ) -> pd.DataFrame:
        """Compute ZLEMA (Zero Lag Exponential Moving Average) for each column of a
        DataFrame. The main idea of ZLEMA is to add a velocity component to the EMA. The
        velocity component is calculated by taking the difference between the current
        value and the value lagged by a certain number of periods (`lag` parameter). The
        velocity component is then multiplied by a constant `K` and added to the current
        value.

        Classic EMA:
            EMA = alpha * X + (1 - alpha) * EMA[-1]

        ZLEMA:
            ZLEMA = alpha * (X + K*(X - X[-lag])) + (1 - alpha) * ZLEMA[-1]
        or more generally:
            ZLEMA = EMA(X + K*(X - X[-lag]))

        Args:
            X (pd.Series|pd.DataFrame): DataFrame which contains the value to compute
                ZLEMA on.
            lag (int): Number of periods to obtain an estimate of the velocity.
            K (float, optional): Velocity coefficient. Defaults to 0.5.
            ewm_kwargs (dict, optional): Arguments for the `ewm` method from pandas.
                Defaults to {"alpha": 0.25, "adjust": False}.
        """
        # Add velocity estimate
        X_adjusted = X + K * (X - X.shift(lag))
        zlema_keys = {"com", "span", "halflife", "alpha"}
        if not zlema_keys.intersection(ewm_kwargs.keys()):
            ewm_kwargs["span"] = 2 * lag + 1
            logging.warning(
                "No parameter for zlema specified, computing default span from lag value"
            )

        # Compute ZLEMA
        return X_adjusted.ewm(**ewm_kwargs)
    

    #
    #["zlema", { "lag": 3, "ewm_kwargs": { "halflife": 14 } }],
    #[mean, {}]
#et pour le sensex:

#    [zlema, {"lag": 5, "ewm_kwargs": {"alpha": 0.25, "adjust": False}}],
#    [mean, {}],
