# Examples

This document provides code snippets and examples of how to implement various parts of the strategy described in the `knowledge_base.md` and `strategia_spiegata.txt` files.

## 1. Initialize the Algorithm

```python
class IntegratedInvestmentStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2020, 12, 31)
        self.SetCash(100000)
        self.AddData(Fred, "GDP")
        self.AddData(Fred, "UNRATE")
        self.AddData(Fred, "CPIAUCSL")
        self.AddData(Fred, "FEDFUNDS")
        self.AddData(Fred, "DGS10")
        self.AddData(Fred, "USSLIND")
        self.macroData = {}
        self.selectedSectors = []
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.selectedStocks = []
        self.mlModel = None
        self.portfolioWeights = {}
        self.historicalDataCache = {}
        self.performanceMetrics = {}
        self.maxPositionSize = 0.05
        self.stopLossPercentage = 0.1
        self.takeProfitPercentage = 0.2
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.AfterMarketOpen("SPY", 30), self.RebalancePortfolio)
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.At(9, 30), self.AnalyzeMacroeconomicData)
        self.lastRebalanceTime = self.Time
        self.marketVolatilityThreshold = 0.02
        self.performanceTracker = {
            'daily_returns': [],
            'portfolio_values': [],
            'drawdowns': [],
            'volatility_window': 252,
            'risk_free_rate': 0.02
        }
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.EndOfDay(), self.TrackPerformance)
        self.volatilityWindow = 20
        self.correlationWindow = 60
        self.anomalyThreshold = 2.0
        self.marketRegimeThresholds = {
            'low_vol': 0.10,
            'high_vol': 0.25
        }
        self.dynamic_risk_params = {
            'base_position_size': 0.05,
            'max_sector_exposure': 0.30,
            'correlation_threshold': 0.7
        }
        self.optimization_params = {
            'target_return': 0.12,
            'risk_aversion': 2.0,
            'max_sector_exposure': 0.30,
            'min_weight': 0.02,
            'rebalance_threshold': 0.1
        }
        self.correlation_matrix = None
        self.sector_exposures = {}
        self.SetBenchmark("SPY")
        self.SetWarmUp(TimeSpan.FromDays(365))
        self.backtest_metrics = {
            'trades': [],
            'daily_pnl': [],
            'positions': {},
            'benchmark_performance': []
        }
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.Settings.LiquidateEnabled = True
        self.indicators = {}
        self.technical_params = {
            'rsi_period': 14,
            'ma_fast': 50,
            'ma_slow': 200,
            'volatility_window': 20
        }
        self.corporate_events = {}
        self.earnings_schedule = {}
        self.Consolidate("SPY", timedelta(days=1), self.OnDataConsolidated)
        self.risk_params = {
            'var_confidence': 0.95,
            'var_window': 252,
            'max_sector_beta': 1.5,
            'min_liquidity_window': 20,
            'max_concentration': 0.25
        }
        self.risk_metrics = {
            'var_history': [],
            'sector_betas': {},
            'tracking_error': [],
            'liquidity_scores': {}
        }
```

## 2. Analyze Macroeconomic Data

```python
def AnalyzeMacroeconomicData(self):
    gdp = self.macroData.get("GDP")
    unemployment = self.macroData.get("UNRATE")
    cpi = self.macroData.get("CPIAUCSL")
    fed_funds = self.macroData.get("FEDFUNDS")
    treasury_yield = self.macroData.get("DGS10")
    leading_index = self.macroData.get("USSLIND")

    indicators = [gdp, unemployment, cpi, fed_funds, treasury_yield, leading_index]
    if all(indicators):
        values = [indicator.Value for indicator in indicators]
        score = (values[0] * 0.2) - (values[1] * 0.2) - (values[2] * 0.1) + (values[5] * 0.5)
        if score > 0.5:
            self.economicRegime = "Expansion"
        elif score < -0.5:
            self.economicRegime = "Recession"
        else:
            self.economicRegime = "Neutral"
    else:
        self.economicRegime = "Neutral"

    self.Debug(f"Economic regime determined: {self.economicRegime}")
```

## 3. Select Sectors

```python
def SelectSectors(self):
    if not hasattr(self, 'economicRegime'):
        self.economicRegime = "Neutral"

    if self.economicRegime == "Expansion":
        self.selectedSectors = ["Technology", "Consumer Discretionary", "Financials"]
    elif self.economicRegime == "Recession":
        self.selectedSectors = ["Utilities", "Consumer Staples", "Healthcare"]
    else:
        self.selectedSectors = ["Industrials", "Materials", "Real Estate"]

    self.Debug(f"Sectors selected based on economic regime ({self.economicRegime}): {self.selectedSectors}")
```

## 4. Select Stocks

```python
def CoarseSelectionFunction(self, coarse):
    sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
    top = sortedByDollarVolume[:1000]
    return [c.Symbol for c in top]

def FineSelectionFunction(self, fine):
    sectorCodes = self.GetSectorCodes(self.selectedSectors)
    selected = []
    for stock in fine:
        if stock.AssetClassification.MorningstarSectorCode in sectorCodes:
            pe_ratio = stock.ValuationRatios.PERatio
            debt_equity = stock.OperationRatios.DebtEquityRatio
            roe = stock.OperationRatios.ROE
            current_ratio = stock.OperationRatios.CurrentRatio
            price_to_book = stock.ValuationRatios.PriceBookRatio

            score = 0
            if pe_ratio is not None and pe_ratio < 20:
                score += 1
            if debt_equity is not None and debt_equity < 0.5:
                score += 1
            if roe is not None and roe > 0.15:
                score += 1
            if current_ratio is not None and current_ratio > 1:
                score += 1
            if price_to_book is not None and price_to_book < 3:
                score += 1

            if score >= 4:
                selected.append(stock.Symbol)

    self.selectedStocks = selected
    self.Log(f"Selected stocks after fundamental scoring: {[str(s) for s in self.selectedStocks]}")
    return self.selectedStocks
```

## 5. Apply Machine Learning

```python
def ApplyMachineLearning(self):
    if not self.selectedStocks:
        return

    try:
        features = []
        targets = []
        symbols = []
        for symbol in self.selectedStocks:
            if symbol in self.historicalDataCache:
                history = self.historicalDataCache[symbol]
            else:
                history = self.History(symbol, 252, Resolution.Daily)
                self.historicalDataCache[symbol] = history

            if not history.empty:
                returns = history['close'].pct_change().dropna()
                mean_return = returns.mean()
                volatility = returns.std()
                pe_ratio = self.Securities[symbol].Fundamentals.ValuationRatios.PERatio
                debt_equity = self.Securities[symbol].Fundamentals.OperationRatios.DebtEquityRatio
                feature = [mean_return, volatility, pe_ratio, debt_equity]
                features.append(feature)
                future_returns = self.History(symbol, 30, Resolution.Daily)['close'].pct_change().dropna()
                target = future_returns.mean()
                targets.append(target)
                symbols.append(symbol)

        if features and targets:
            selector = SelectKBest(f_regression, k='all')
            features_selected = selector.fit_transform(features, targets)

            model = LinearRegression()
            cv_scores = cross_val_score(model, features_selected, targets, cv=5)
            self.Debug(f"Cross-validation scores: {cv_scores}")

            model.fit(features_selected, targets)
            self.mlModel = model

    except Exception as e:
        self.Error(f"Error in ApplyMachineLearning: {e}")
```

## 6. Optimize Portfolio

```python
def OptimizePortfolio(self):
    if not self.mlModel or not self.selectedStocks:
        return

    try:
        returns_data = {}
        for symbol in self.selectedStocks:
            if symbol in self.historicalDataCache:
                history = self.historicalDataCache[symbol]
            else:
                history = self.History(symbol, 252, Resolution.Daily)
                self.historicalDataCache[symbol] = history

            if not history.empty:
                returns_data[str(symbol)] = history['close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        mu = returns_df.mean() * 252
        Sigma = returns_df.cov() * 252
        self.correlation_matrix = returns_df.corr()

        def objective(weights):
            portfolio_return = np.sum(weights * mu)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            return -portfolio_return/portfolio_risk

        n_assets = len(self.selectedStocks)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x - self.optimization_params['min_weight']},
            {'type': 'ineq', 'fun': lambda x: self.optimization_params['max_sector_exposure'] - self.calculate_sector_exposures(x)}
        ]

        bounds = tuple((self.optimization_params['min_weight'], self.maxPositionSize) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        risk_metrics = self.CalculatePortfolioRisk()
        if risk_metrics.get('VaR'):
            var_constraint = {'type': 'ineq', 'fun': lambda x: -self.CalculateValueAtRisk(x) + risk_metrics['VaR']}
            constraints.append(var_constraint)

        liquidity_constraint = {'type': 'ineq', 'fun': lambda x: self.CheckLiquidityConstraints(x)}
        constraints.append(liquidity_constraint)

        if result.success:
            self.portfolioWeights = {
                symbol: weight for symbol, weight in zip(self.selectedStocks, result.x)
            }
            self.implement_portfolio_changes()
            self.log_portfolio_metrics(mu, Sigma, result.x)
        else:
            self.Error("Portfolio optimization failed")

    except Exception as e:
        self.Error(f"Error in OptimizePortfolio: {e}")
```

## 7. Track Performance

```python
def TrackPerformance(self):
    try:
        current_value = self.Portfolio.TotalPortfolioValue
        if len(self.performanceTracker['portfolio_values']) > 0:
            prev_value = self.performanceTracker['portfolio_values'][-1]
            daily_return = (current_value - prev_value) / prev_value
            self.performanceTracker['daily_returns'].append(daily_return)

        self.performanceTracker['portfolio_values'].append(current_value)

        if len(self.performanceTracker['daily_returns']) > 20:
            returns_array = np.array(self.performanceTracker['daily_returns'])
            excess_returns = returns_array - (self.performanceTracker['risk_free_rate']/252)
            sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns_array)
            peak = np.maximum.accumulate(self.performanceTracker['portfolio_values'])
            drawdown = (peak - self.performanceTracker['portfolio_values']) / peak
            max_drawdown = np.max(drawdown)
            vol = np.std(returns_array) * np.sqrt(252)

            self.Log(f"Performance Metrics - Date: {self.Time}")
            self.Log(f"Sharpe Ratio: {sharpe:.2f}")
            self.Log(f"Max Drawdown: {max_drawdown:.2%}")
            self.Log(f"Annualized Volatility: {vol:.2%}")

            self.Plot("Risk Metrics", "Sharpe Ratio", sharpe)
            self.Plot("Risk Metrics", "Max Drawdown", max_drawdown)
            self.Plot("Risk Metrics", "Volatility", vol)

    except Exception as e:
        self.Error(f"Error in TrackPerformance: {e}")
```

## 8. Rebalance Portfolio

```python
def RebalancePortfolio(self):
    time_since_last_rebalance = (self.Time - self.lastRebalanceTime).days
    if time_since_last_rebalance >= 30 or self.forceRebalance:
        self.AnalyzeMacroeconomicData()
        self.SelectSectors()
        self.ApplyMachineLearning()
        self.OptimizePortfolio()
        self.lastRebalanceTime = self.Time
        self.forceRebalance = False
```

## 9. Handle Corporate Events

```python
def HandleCorporateEvents(self, data):
    try:
        for symbol in self.Portfolio.Keys:
            if symbol in data and hasattr(data[symbol], 'HasEarnings'):
                if data[symbol].HasEarnings:
                    self.Log(f"Earnings event detected for {symbol}")
                    self.SetHoldings(symbol, self.Portfolio[symbol].Weight * 0.5)

            if symbol in data and hasattr(data[symbol], 'HasDividend'):
                if data[symbol].HasDividend:
                    self.Log(f"Dividend event detected for {symbol}")
                    self.corporate_events[symbol] = {
                        'type': 'dividend',
                        'amount': data[symbol].Dividend,
                        'date': self.Time
                    }

    except Exception as e:
        self.Error(f"Error in HandleCorporateEvents: {e}")
```

## 10. Calculate Portfolio Risk

```python
def CalculatePortfolioRisk(self):
    try:
        portfolio_stats = {}
        portfolio_returns = np.array(self.performanceTracker['daily_returns'])
        if len(portfolio_returns) > self.risk_params['var_window']:
            var = self.CalculateValueAtRisk(portfolio_returns)
            self.risk_metrics['var_history'].append(var)
            portfolio_stats['VaR'] = var

        self.CalculateSectorBetas()
        self.CalculateTrackingError()
        self.MonitorLiquidity()

        return portfolio_stats

    except Exception as e:
        self.Error(f"Error in CalculatePortfolioRisk: {e}")
        return {}
```

## 11. Calculate Value at Risk

```python
def CalculateValueAtRisk(self, returns, confidence=None):
    if confidence is None:
        confidence = self.risk_params['var_confidence']

    try:
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[index]
        return var

    except Exception as e:
        self.Error(f"Error in CalculateValueAtRisk: {e}")
        return None
```

## 12. Calculate Sector Betas

```python
def CalculateSectorBetas(self):
    try:
        spy_history = self.History(self.Symbol("SPY"), 252, Resolution.Daily)
        spy_returns = spy_history['close'].pct_change().dropna()

        for sector in set(self.sector_exposures.keys()):
            sector_symbols = [s for s in self.Portfolio.Keys if self.Securities[s].Fundamentals.AssetClassification.MorningstarSectorCode == sector]

            if sector_symbols:
                sector_returns = pd.DataFrame()
                for symbol in sector_symbols:
                    history = self.History(symbol, 252, Resolution.Daily)
                    if not history.empty:
                        sector_returns[str(symbol)] = history['close'].pct_change()

                sector_portfolio_returns = sector_returns.mean(axis=1)
                beta = self.CalculateBeta(sector_portfolio_returns, spy_returns)
                self.risk_metrics['sector_betas'][sector] = beta

                if beta > self.risk_params['max_sector_beta']:
                    self.Log(f"Warning: High beta ({beta:.2f}) detected in sector {sector}")

    except Exception as e:
        self.Error(f"Error in CalculateSectorBetas: {e}")
```

## 13. Monitor Liquidity

```python
def MonitorLiquidity(self):
    try:
        for symbol in self.Portfolio.Keys:
            history = self.History(symbol, self.risk_params['min_liquidity_window'], Resolution.Daily)
            if not history.empty:
                avg_volume = history['volume'].mean()
                avg_price = history['close'].mean()
                daily_liquidity = avg_volume * avg_price

                self.risk_metrics['liquidity_scores'][symbol] = daily_liquidity

                position_value = self.Portfolio[symbol].HoldingsValue
                if position_value > daily_liquidity * 0.1:
                    self.Log(f"Warning: Potential liquidity issue with {symbol}")

    except Exception as e:
        self.Error(f"Error in MonitorLiquidity: {e}")
```

## 14. Check Liquidity Constraints

```python
def CheckLiquidityConstraints(self, weights):
    try:
        liquidity_scores = []
        for symbol, weight in zip(self.selectedStocks, weights):
            if symbol in self.risk_metrics['liquidity_scores']:
                position_value = self.Portfolio.TotalPortfolioValue * weight
                liquidity_score = position_value / self.risk_metrics['liquidity_scores'][symbol]
                liquidity_scores.append(liquidity_score)

        return -max(liquidity_scores) + 0.1

    except Exception as e:
        self.Error(f"Error in CheckLiquidityConstraints: {e}")
        return 0
```
