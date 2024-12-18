from QuantConnect.Data.Custom.Fred import *
from QuantConnect.Algorithm import QCAlgorithm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy import stats
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import TradeBar
from QuantConnect.Data.Fundamental import FineFundamental
from QuantConnect.Risk import RiskMetrics
from QuantConnect.Algorithm.Framework.Risk import RiskManagementModel
from QuantConnect.Algorithm.Framework.Portfolio import PortfolioTarget

class IntegratedInvestmentStrategy(QCAlgorithm):

    def Initialize(self):
        # Impostare la data di inizio e fine
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2020, 12, 31)
        # Impostare il capitale iniziale
        self.SetCash(100000)
        # Sottoscrivere gli indicatori macroeconomici dal database FRED
        self.AddEquity("SPY")
        self.AddData(Fred, "UNRATE")
        self.AddData(Fred, "CPIAUCSL")
        # Sottoscrivere indicatori macroeconomici aggiuntivi dal database FRED
        self.AddData(Fred, "FEDFUNDS")    # Tasso dei fondi federali
        self.AddData(Fred, "DGS10")       # Tasso Treasury a 10 anni
        self.AddData(Fred, "USSLIND")     # Leading Index
        # Inizializzare variabili necessarie
        self.macroData = {}
        self.selectedSectors = []
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.selectedStocks = []
        self.mlModel = None
        self.portfolioWeights = {}
        self.historicalDataCache = {}
        self.performanceMetrics = {}

        # Impostare limiti per la gestione del rischio
        self.maxPositionSize = 0.05  # 5% del portafoglio per singolo titolo
        self.stopLossPercentage = 0.1  # Stop-loss al 10% di perdita
        self.takeProfitPercentage = 0.2  # Take-profit al 20% di guadagno

        # Pianificare il ribilanciamento del portafoglio all'inizio di ogni mese
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.AfterMarketOpen("SPY", 30), self.RebalancePortfolio)

        # Pianificare l'aggiornamento dell'analisi macroeconomica ogni settimana
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.At(9, 30), self.AnalyzeMacroeconomicData)

        # Variabile per controllare la frequenza di ribilanciamento
        self.lastRebalanceTime = self.Time
        # Soglia per la volatilità del mercato
        self.marketVolatilityThreshold = 0.02  # 2% volatilità giornaliera dell'indice di riferimento

        # Inizializzare variabili per il monitoraggio delle performance
        self.performanceTracker = {
            'daily_returns': [],
            'portfolio_values': [],
            'drawdowns': [],
            'volatility_window': 252,  # Un anno di trading
            'risk_free_rate': 0.02     # 2% annuo
        }
        
        # Pianificare il reporting giornaliero
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.EndOfDay(), 
                        self.TrackPerformance)

        # Parametri per il monitoraggio degli eventi di mercato
        self.volatilityWindow = 20
        self.correlationWindow = 60
        self.anomalyThreshold = 2.0  # Standard deviations
        self.marketRegimeThresholds = {
            'low_vol': 0.10,
            'high_vol': 0.25
        }
        
        # Parametri dinamici per la gestione del rischio
        self.dynamic_risk_params = {
            'base_position_size': 0.05,
            'max_sector_exposure': 0.30,
            'correlation_threshold': 0.7
        }

        # Parametri per l'ottimizzazione del portafoglio
        self.optimization_params = {
            'target_return': 0.12,  # Target di rendimento annuale
            'risk_aversion': 2.0,   # Coefficiente di avversione al rischio
            'max_sector_exposure': 0.30,  # Esposizione massima per settore
            'min_weight': 0.02,     # Peso minimo per posizione
            'rebalance_threshold': 0.1  # Soglia di ribilanciamento (10% di deriva)
        }
        
        # Dizionario per tracciare le correlazioni
        self.correlation_matrix = None
        self.sector_exposures = {}

        # Impostare il benchmark
        self.SetBenchmark("SPY")
        
        # Periodo di warm-up per il training dei modelli
        self.SetWarmUp(TimeSpan.FromDays(365))
        
        # Parametri di backtesting
        self.backtest_metrics = {
            'trades': [],
            'daily_pnl': [],
            'positions': {},
            'benchmark_performance': []
        }
        
        # Impostazioni per la gestione degli ordini
        self.Settings.FreePortfolioValuePercentage = 0.05  # Mantiene 5% in cash
        self.Settings.LiquidateEnabled = True  # Abilita liquidazione automatica

        # Dizionario per gli indicatori tecnici
        self.indicators = {}
        
        # Parametri per gli indicatori tecnici
        self.technical_params = {
            'rsi_period': 14,
            'ma_fast': 50,
            'ma_slow': 200,
            'volatility_window': 20
        }
        
        # Tracciamento degli eventi corporate
        self.corporate_events = {}
        
        # Schedule degli earnings
        self.earnings_schedule = {}
        
        # Aggiungere il consolidatore per i dati giornalieri
        self.Consolidate("SPY", timedelta(days=1), self.OnDataConsolidated)

        # Parametri per la gestione del rischio avanzata
        self.risk_params = {
            'var_confidence': 0.95,
            'var_window': 252,
            'max_sector_beta': 1.5,
            'min_liquidity_window': 20,
            'max_concentration': 0.25  # 25% massimo per settore
        }
        
        # Dizionario per il tracking delle metriche di rischio
        self.risk_metrics = {
            'var_history': [],
            'sector_betas': {},
            'tracking_error': [],
            'liquidity_scores': {}
        }

        pass

    def OnData(self, data):
        # Gestire i dati in arrivo
        for indicator in ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10", "USSLIND"]:
            if data.ContainsKey(indicator):
                self.macroData[indicator] = data[indicator]

        # Monitorare le posizioni aperte per applicare stop-loss e take-profit
        for symbol, position in self.Portfolio.items():
            if position.Invested:
                unrealizedProfitPercent = position.UnrealizedProfitPercent
                if unrealizedProfitPercent <= -self.stopLossPercentage:
                    self.Liquidate(symbol)
                    self.Log(f"Stop-loss triggered for {symbol}")
                elif unrealizedProfitPercent >= self.takeProfitPercentage:
                    self.Liquidate(symbol)
                    self.Log(f"Take-profit triggered for {symbol}")

        # Monitorare gli eventi di mercato inattesi
        self.MonitorMarketEvents()

        # Rilevare eventi di mercato e aggiustare i parametri
        market_regime = self.DetectMarketEvents()
        if market_regime != getattr(self, 'previous_market_regime', None):
            self.Log(f"Market regime changed to: {market_regime}")
            self.previous_market_regime = market_regime
            self.AdjustRiskParameters(market_regime)

        # Gestire gli eventi corporate
        self.HandleCorporateEvents(data)

        pass

    def MonitorMarketEvents(self):
        # Monitorare la volatilità del mercato o altri indicatori di rischio
        spy_history = self.History(["SPY"], 30, Resolution.Daily)
        if not spy_history.empty:
            spy_returns = spy_history["close"].pct_change().dropna()
            market_volatility = spy_returns.std()
            self.Debug(f"Market Volatility: {market_volatility}")

            if market_volatility > self.marketVolatilityThreshold:
                # Adeguare il portafoglio riducendo l'esposizione
                self.SetHoldings("SPY", 0)
                self.Liquidate()
                self.Debug("High market volatility detected, liquidating positions.")
                # Aggiornare l'analisi macro e il portafoglio
                self.RebalancePortfolio()

    def DetectMarketEvents(self):
        """Rileva eventi di mercato significativi e anomalie"""
        try:
            # Ottenere i dati di mercato recenti
            spy_history = self.History(["SPY"], self.correlationWindow, Resolution.Daily)
            if spy_history.empty:
                return

            # Calcolare le statistiche di mercato
            returns = spy_history['close'].pct_change().dropna()
            rolling_vol = returns.rolling(window=self.volatilityWindow).std()
            current_vol = rolling_vol.iloc[-1]
            
            # Determinare il regime di volatilità
            if current_vol < self.marketRegimeThresholds['low_vol']:
                self.market_regime = "Low Volatility"
            elif current_vol > self.marketRegimeThresholds['high_vol']:
                self.market_regime = "High Volatility"
            else:
                self.market_regime = "Normal Volatility"

            # Rilevare anomalie statistiche
            z_score = (returns.iloc[-1] - returns.mean()) / returns.std()
            if abs(z_score) > self.anomalyThreshold:
                self.Log(f"Market anomaly detected: z-score = {z_score:.2f}")
                self.AdjustRiskParameters(z_score)

            return self.market_regime

        except Exception as e:
            self.Error(f"Error in DetectMarketEvents: {e}")
            return "Normal Volatility"

    def AdjustRiskParameters(self, market_stress_level):
        """Aggiusta dinamicamente i parametri di rischio"""
        try:
            # Adattare i parametri di stop-loss e take-profit
            base_stop_loss = self.stopLossPercentage
            volatility_adjustment = abs(market_stress_level) / 10.0
            
            if market_stress_level > 0:
                # Mercato più rischioso: stop più stretti
                self.stopLossPercentage = base_stop_loss * (1 - volatility_adjustment)
                self.takeProfitPercentage = self.takeProfitPercentage * (1 + volatility_adjustment)
            else:
                # Mercato meno rischioso: stop più larghi
                self.stopLossPercentage = base_stop_loss * (1 + volatility_adjustment)
                self.takeProfitPercentage = self.takeProfitPercentage * (1 - volatility_adjustment)

            # Aggiustare la dimensione massima delle posizioni
            self.maxPositionSize = self.dynamic_risk_params['base_position_size'] * \
                                 (1 - abs(market_stress_level) / 5.0)

            self.Log(f"Risk parameters adjusted - Stop Loss: {self.stopLossPercentage:.2%}, " +
                    f"Take Profit: {self.takeProfitPercentage:.2%}, " +
                    f"Max Position Size: {self.maxPositionSize:.2%}")

        except Exception as e:
            self.Error(f"Error in AdjustRiskParameters: {e}")

    def AnalyzeMacroeconomicData(self):
        # Analizzare gli indicatori macroeconomici per determinare il regime economico
        gdp = self.macroData.get("GDP")
        unemployment = self.macroData.get("UNRATE")
        cpi = self.macroData.get("CPIAUCSL")
        fed_funds = self.macroData.get("FEDFUNDS")
        treasury_yield = self.macroData.get("DGS10")
        leading_index = self.macroData.get("USSLIND")

        # Implementare una logica più avanzata per determinare il regime economico
        indicators = [gdp, unemployment, cpi, fed_funds, treasury_yield, leading_index]
        if all(indicators):
            values = [indicator.Value for indicator in indicators]
            # Calcolare uno score per il regime economico
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

    def SelectSectors(self):
        # Selezionare i settori basandosi sull'analisi macroeconomica
        if not hasattr(self, 'economicRegime'):
            self.economicRegime = "Neutral"

        if self.economicRegime == "Expansion":
            self.selectedSectors = ["Technology", "Consumer Discretionary", "Financials"]
        elif self.economicRegime == "Recession":
            self.selectedSectors = ["Utilities", "Consumer Staples", "Healthcare"]
        else:
            self.selectedSectors = ["Industrials", "Materials", "Real Estate"]

        self.Debug(f"Sectors selected based on economic regime ({self.economicRegime}): {self.selectedSectors}")
        # ...eventuale codice aggiuntivo per elaborare i settori selezionati...
        pass

    def SelectStocks(self):
        # La selezione dei titoli è gestita dalle funzioni di selezione dell'universo
        pass

    def CoarseSelectionFunction(self, coarse):
        # Filtra i primi 1000 titoli per volume di scambio
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        top = sortedByDollarVolume[:1000]
        return [c.Symbol for c in top]

    def FineSelectionFunction(self, fine):
        # Migliorare l'analisi fondamentale con più metriche
        sectorCodes = self.GetSectorCodes(self.selectedSectors)
        selected = []
        for stock in fine:
            if stock.AssetClassification.MorningstarSectorCode in sectorCodes:
                # Applicare criteri di analisi fondamentale avanzati
                pe_ratio = stock.ValuationRatios.PERatio
                debt_equity = stock.OperationRatios.DebtEquityRatio
                roe = stock.OperationRatios.ROE
                current_ratio = stock.OperationRatios.CurrentRatio
                price_to_book = stock.ValuationRatios.PriceBookRatio

                # Calcolare un punteggio basato su diversi criteri
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

    def GetSectorCodes(self, sectors):
        # Mappa i nomi dei settori ai codici Morningstar
        sectorCodeMap = {
            "Basic Materials": MorningstarSectorCode.BasicMaterials,
            "Consumer Cyclical": MorningstarSectorCode.ConsumerCyclical,
            "Financial Services": MorningstarSectorCode.FinancialServices,
            "Real Estate": MorningstarSectorCode.RealEstate,
            "Consumer Defensive": MorningstarSectorCode.ConsumerDefensive,
            "Healthcare": MorningstarSectorCode.Healthcare,
            "Utilities": MorningstarSectorCode.Utilities,
            "Communication Services": MorningstarSectorCode.CommunicationServices,
            "Energy": MorningstarSectorCode.Energy,
            "Industrials": MorningstarSectorCode.Industrials,
            "Technology": MorningstarSectorCode.Technology
        }
        return [sectorCodeMap[sector] for sector in sectors if sector in sectorCodeMap]

    def ApplyMachineLearning(self):
        # Assicurarsi che il machine learning sia utilizzato come supporto
        if not self.selectedStocks:
            return

        try:
            # Preparare i dati per l'addestramento del modello
            features = []
            targets = []
            symbols = []
            for symbol in self.selectedStocks:
                if symbol in self.historicalDataCache:
                    history = self.historicalDataCache[symbol]
                else:
                    history = self.History([symbol], 252, Resolution.Daily)
                    self.historicalDataCache[symbol] = history

                if not history.empty:
                    # Estrarre caratteristiche come rendimenti e volatilità
                    returns = history['close'].pct_change().dropna()
                    mean_return = returns.mean()
                    volatility = returns.std()
                    pe_ratio = self.Securities[symbol].Fundamentals.ValuationRatios.PERatio
                    debt_equity = self.Securities[symbol].Fundamentals.OperationRatios.DebtEquityRatio
                    # Creare il vettore delle caratteristiche
                    feature = [mean_return, volatility, pe_ratio, debt_equity]
                    features.append(feature)
                    # Utilizzare il rendimento futuro come target
                    future_returns = self.History([symbol], 30, Resolution.Daily)['close'].pct_change().dropna()
                    target = future_returns.mean()
                    targets.append(target)
                    symbols.append(symbol)

            if features and targets:
                # Selezione delle caratteristiche
                selector = SelectKBest(f_regression, k='all')
                features_selected = selector.fit_transform(features, targets)

                # Utilizzare un modello lineare per evitare overfitting
                model = LinearRegression()
                cv_scores = cross_val_score(model, features_selected, targets, cv=5)
                self.Debug(f"Cross-validation scores: {cv_scores}")

                # Addestrare il modello
                model.fit(features_selected, targets)
                self.mlModel = model

        except Exception as e:
            self.Error(f"Error in ApplyMachineLearning: {e}")

    def OptimizePortfolio(self):
        """Implementa l'ottimizzazione di Markowitz con vincoli personalizzati"""
        if not self.mlModel or not self.selectedStocks:
            return

        try:
            # Preparare i dati per l'ottimizzazione
            returns_data = {}
            for symbol in self.selectedStocks:
                if symbol in self.historicalDataCache:
                    history = self.historicalDataCache[symbol]
                else:
                    history = self.History([symbol], 252, Resolution.Daily)
                    self.historicalDataCache[symbol] = history

                if not history.empty:
                    returns_data[str(symbol)] = history['close'].pct_change().dropna()

            # Creare il DataFrame dei rendimenti
            returns_df = pd.DataFrame(returns_data)
            
            # Calcolare media e covarianza
            mu = returns_df.mean() * 252  # Rendimenti annualizzati
            Sigma = returns_df.cov() * 252  # Covarianza annualizzata
            
            # Aggiornare la matrice di correlazione
            self.correlation_matrix = returns_df.corr()

            # Definire la funzione obiettivo (Sharpe Ratio negativo da minimizzare)
            def objective(weights):
                portfolio_return = np.sum(weights * mu)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
                return -portfolio_return/portfolio_risk

            # Vincoli
            n_assets = len(self.selectedStocks)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Somma pesi = 1
                {'type': 'ineq', 'fun': lambda x: x - self.optimization_params['min_weight']},  # Peso minimo
                {'type': 'ineq', 'fun': lambda x: self.optimization_params['max_sector_exposure'] - 
                 self.calculate_sector_exposures(x)}  # Vincolo settoriale
            ]

            # Bounds per i pesi
            bounds = tuple((self.optimization_params['min_weight'], 
                          self.maxPositionSize) for _ in range(n_assets))

            # Ottimizzazione
            initial_weights = np.array([1/n_assets] * n_assets)
            result = minimize(objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            # Aggiungere vincoli di rischio
            risk_metrics = self.CalculatePortfolioRisk()
            
            if risk_metrics.get('VaR'):
                var_constraint = {'type': 'ineq', 
                                'fun': lambda x: -self.CalculateValueAtRisk(x) + risk_metrics['VaR']}
                constraints.append(var_constraint)
            
            # Aggiungere vincoli di liquidità
            liquidity_constraint = {'type': 'ineq',
                                  'fun': lambda x: self.CheckLiquidityConstraints(x)}
            constraints.append(liquidity_constraint)

            if result.success:
                # Aggiornare i pesi del portafoglio
                self.portfolioWeights = {
                    symbol: weight for symbol, weight in zip(self.selectedStocks, result.x)
                }
                
                # Implementare il portafoglio solo se necessario (controllo della deriva)
                self.implement_portfolio_changes()
                
                # Loggare le metriche del portafoglio ottimizzato
                self.log_portfolio_metrics(mu, Sigma, result.x)
            else:
                self.Error("Portfolio optimization failed")

        except Exception as e:
            self.Error(f"Error in OptimizePortfolio: {e}")

    def calculate_sector_exposures(self, weights):
        """Calcola l'esposizione per settore"""
        sector_exposures = {}
        for symbol, weight in zip(self.selectedStocks, weights):
            sector = self.Securities[symbol].Fundamentals.AssetClassification.MorningstarSectorCode
            sector_exposures[sector] = sector_exposures.get(sector, 0) + weight
        return max(sector_exposures.values())

    def implement_portfolio_changes(self):
        """Implementa i cambiamenti nel portafoglio con controlli di liquidità"""
        current_holdings = {str(x.Symbol): x.HoldingsValue/self.Portfolio.TotalPortfolioValue 
                          for x in self.Portfolio.Values if x.Invested}
        
        ordered_symbols = sorted(self.portfolioWeights.items(), 
                               key=lambda x: abs(x[1] - current_holdings.get(str(x[0]), 0)),
                               reverse=True)
        
        for symbol, target_weight in ordered_symbols:
            current_weight = current_holdings.get(str(symbol), 0)
            weight_drift = abs(target_weight - current_weight)
            
            if weight_drift > self.optimization_params['rebalance_threshold']:
                if self.ValidateAndPlaceOrder(symbol, target_weight):
                    self.Log(f"Rebalancing {symbol}: {current_weight:.2%} -> {target_weight:.2%}")

    def RebalancePortfolio(self):
        # Riequilibrare il portafoglio periodicamente o in risposta a eventi di mercato

        # Controllare se è il momento di ribilanciare
        time_since_last_rebalance = (self.Time - self.lastRebalanceTime).days
        if time_since_last_rebalance >= 30 or self.forceRebalance:
            # Aggiornare l'analisi macroeconomica
            self.AnalyzeMacroeconomicData()

            # Selezionare i settori in base al nuovo regime economico
            self.SelectSectors()

            # Applicare il machine learning per affinare le stime
            self.ApplyMachineLearning()

            # Ottimizzare e aggiornare il portafoglio
            self.OptimizePortfolio()

            # Aggiornare il timestamp dell'ultimo ribilanciamento
            self.lastRebalanceTime = self.Time
            self.forceRebalance = False

    def TrackPerformance(self):
        """Calcola e registra le metriche di performance giornaliere"""
        try:
            # Calcolare il rendimento giornaliero
            current_value = self.Portfolio.TotalPortfolioValue
            if len(self.performanceTracker['portfolio_values']) > 0:
                prev_value = self.performanceTracker['portfolio_values'][-1]
                daily_return = (current_value - prev_value) / prev_value
                self.performanceTracker['daily_returns'].append(daily_return)
            
            self.performanceTracker['portfolio_values'].append(current_value)
            
            # Calcolare le metriche se abbiamo abbastanza dati
            if len(self.performanceTracker['daily_returns']) > 20:
                # Sharpe Ratio
                returns_array = np.array(self.performanceTracker['daily_returns'])
                excess_returns = returns_array - (self.performanceTracker['risk_free_rate']/252)
                sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns_array)
                
                # Maximum Drawdown
                peak = np.maximum.accumulate(self.performanceTracker['portfolio_values'])
                drawdown = (peak - self.performanceTracker['portfolio_values']) / peak
                max_drawdown = np.max(drawdown)
                
                # Volatilità annualizzata
                vol = np.std(returns_array) * np.sqrt(252)
                
                # Log delle metriche
                self.Log(f"Performance Metrics - Date: {self.Time}")
                self.Log(f"Sharpe Ratio: {sharpe:.2f}")
                self.Log(f"Max Drawdown: {max_drawdown:.2%}")
                self.Log(f"Annualized Volatility: {vol:.2%}")
                
                # Plot delle metriche
                self.Plot("Risk Metrics", "Sharpe Ratio", sharpe)
                self.Plot("Risk Metrics", "Max Drawdown", max_drawdown)
                self.Plot("Risk Metrics", "Volatility", vol)

        except Exception as e:
            self.Error(f"Error in TrackPerformance: {e}")

    def OnEndOfDay(self, symbol):
        """Esegue analisi di fine giornata e genera report"""
        # Analisi della composizione del portafoglio
        total_value = self.Portfolio.TotalPortfolioValue
        for kvp in self.Portfolio:
            if kvp.Value.Invested:
                position = kvp.Value
                weight = position.HoldingsValue / total_value
                self.Log(f"Position: {position.Symbol}, " +
                        f"Weight: {weight:.2%}, " +
                        f"Unrealized P/L: {position.UnrealizedProfitPercent:.2%}")
        
        # Pulizia della cache se necessario
        self.CleanHistoricalDataCache()

    def OnSecuritiesChanged(self, changes):
        # Gestire i cambiamenti nei titoli in portafoglio
        for removed in changes.RemovedSecurities:
            if removed.Symbol in self.Portfolio:
                self.Liquidate(removed.Symbol)
        pass

    def OnRiskManagement(self):
        # Monitorare condizioni per la gestione del rischio
        # ...eventuale codice per gestire rischi specifici...

        pass

    def CleanHistoricalDataCache(self):
        # Pulire la cache dei dati storici per liberare memoria
        if len(self.historicalDataCache) > 500:
            self.historicalDataCache.clear()
            self.Debug("Historical data cache cleared to free memory.")

    def OnOrderEvent(self, orderEvent):
        """Gestisce gli eventi degli ordini"""
        if orderEvent.Status == OrderStatus.Filled:
            self.backtest_metrics['trades'].append({
                'symbol': orderEvent.Symbol,
                'quantity': orderEvent.FillQuantity,
                'price': orderEvent.FillPrice,
                'time': self.Time,
                'direction': "BUY" if orderEvent.FillQuantity > 0 else "SELL"
            })
            self.Log(f"Order filled - {orderEvent.Symbol}: {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")

    def ValidateAndPlaceOrder(self, symbol, target_weight):
        """Valida e piazza un ordine considerando la liquidità"""
        try:
            # Verificare la liquidità disponibile
            price = self.Securities[symbol].Price
            portfolio_value = self.Portfolio.TotalPortfolioValue
            current_holding = self.Portfolio[symbol].Quantity
            target_quantity = (target_weight * portfolio_value) / price
            quantity_difference = target_quantity - current_holding

            # Verificare se abbiamo abbastanza liquidità
            required_cash = abs(quantity_difference) * price
            if required_cash > self.Portfolio.Cash * 0.95:  # Mantiene un buffer del 5%
                self.Debug(f"Insufficient liquidity for {symbol} order. Required: {required_cash}, Available: {self.Portfolio.Cash}")
                return False

            # Piazzare l'ordine
            if abs(quantity_difference) > 0:
                ticket = self.MarketOrder(symbol, quantity_difference)
                if ticket.Status == OrderStatus.Invalid:
                    self.Debug(f"Invalid order for {symbol}: {ticket.Message}")
                    return False
                return True

        except Exception as e:
            self.Error(f"Error in ValidateAndPlaceOrder: {e}")
            return False

    def OnEndOfAlgorithm(self):
        """Genera report finale del backtest"""
        try:
            # Calcolare statistiche complessive
            total_trades = len(self.backtest_metrics['trades'])
            winning_trades = sum(1 for trade in self.backtest_metrics['trades'] 
                               if trade['price'] > trade['price'])
            
            # Calcolare metriche di performance vs benchmark
            algorithm_return = (self.Portfolio.TotalPortfolioValue / self.StartingCapital) - 1
            benchmark_return = (self.Benchmark.Evaluate(self.Time) / 
                              self.Benchmark.Evaluate(self.StartDate)) - 1
            
            # Logging delle statistiche finali
            self.Log(f"\nBacktest Summary:")
            self.Log(f"Total Trades: {total_trades}")
            self.Log(f"Win Rate: {(winning_trades/total_trades if total_trades > 0 else 0):.2%}")
            self.Log(f"Algorithm Return: {algorithm_return:.2%}")
            self.Log(f"Benchmark Return: {benchmark_return:.2%}")
            self.Log(f"Alpha: {algorithm_return - benchmark_return:.2%}")

        except Exception as e:
            self.Error(f"Error in OnEndOfAlgorithm: {e}")

    def InitializeIndicators(self, symbol):
        """Inizializza gli indicatori tecnici per un simbolo"""
        if symbol not in self.indicators:
            self.indicators[symbol] = {
                'RSI': self.RSI(symbol, self.technical_params['rsi_period']),
                'SMA_fast': self.SMA(symbol, self.technical_params['ma_fast']),
                'SMA_slow': self.SMA(symbol, self.technical_params['ma_slow']),
                'BB': self.BB(symbol, 20, 2),
                'MACD': self.MACD(symbol, 12, 26, 9)
            }

    def OnDataConsolidated(self, sender, bar):
        """Gestisce i dati consolidati giornalieri"""
        if self.indicators.get(bar.Symbol) is None:
            self.InitializeIndicators(bar.Symbol)
        
        indicators = self.indicators[bar.Symbol]
        
        # Analizzare i segnali tecnici
        if self.AreIndicatorsReady(indicators):
            signal = self.AnalyzeTechnicalSignals(bar.Symbol, indicators)
            self.ProcessTechnicalSignal(bar.Symbol, signal)

    def AreIndicatorsReady(self, indicators):
        """Verifica se gli indicatori sono pronti"""
        return all(indicator.IsReady for indicator in indicators.values())

    def AnalyzeTechnicalSignals(self, symbol, indicators):
        """Analizza i segnali tecnici per generare segnali di trading"""
        try:
            rsi = indicators['RSI'].Current.Value
            fast_ma = indicators['SMA_fast'].Current.Value
            slow_ma = indicators['SMA_slow'].Current.Value
            macd = indicators['MACD']
            
            signal = {
                'RSI_oversold': rsi < 30,
                'RSI_overbought': rsi > 70,
                'Golden_Cross': fast_ma > slow_ma and self.Securities[symbol].Price > fast_ma,
                'Death_Cross': fast_ma < slow_ma and self.Securities[symbol].Price < fast_ma,
                'MACD_bullish': macd.Current.Value > macd.Signal.Current.Value,
                'MACD_bearish': macd.Current.Value < macd.Signal.Current.Value
            }
            
            return signal

        except Exception as e:
            self.Error(f"Error in AnalyzeTechnicalSignals: {e}")
            return None

    def ProcessTechnicalSignal(self, symbol, signal):
        """Processa i segnali tecnici e li integra con l'analisi macro"""
        if signal is None:
            return

        try:
            # Combinare segnali tecnici con il regime macroeconomico
            if self.economicRegime == "Expansion":
                if signal['Golden_Cross'] and signal['MACD_bullish'] and not signal['RSI_overbought']:
                    self.Log(f"Bullish technical setup for {symbol} in expansion regime")
                    self.SetHoldings(symbol, self.maxPositionSize)
            
            elif self.economicRegime == "Recession":
                if signal['RSI_oversold'] and signal['MACD_bullish']:
                    self.Log(f"Potential reversal setup for {symbol} in recession regime")
                    self.SetHoldings(symbol, self.maxPositionSize * 0.5)

        except Exception as e:
            self.Error(f"Error in ProcessTechnicalSignal: {e}")

    def HandleCorporateEvents(self, data):
        """Gestisce gli eventi corporate come earnings e dividendi"""
        try:
            for symbol in self.Portfolio.Keys:
                if symbol in data and hasattr(data[symbol], 'HasEarnings'):
                    if data[symbol].HasEarnings:
                        self.Log(f"Earnings event detected for {symbol}")
                        # Ridurre l'esposizione prima degli earnings
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

    def CalculatePortfolioRisk(self):
        """Calcola le metriche di rischio del portafoglio"""
        try:
            portfolio_stats = {}
            
            # Calcolare il VaR del portafoglio
            portfolio_returns = np.array(self.performanceTracker['daily_returns'])
            if len(portfolio_returns) > self.risk_params['var_window']:
                var = self.CalculateValueAtRisk(portfolio_returns)
                self.risk_metrics['var_history'].append(var)
                portfolio_stats['VaR'] = var
            
            # Calcolare i beta settoriali
            self.CalculateSectorBetas()
            
            # Calcolare il tracking error
            self.CalculateTrackingError()
            
            # Monitorare la liquidità
            self.MonitorLiquidity()
            
            return portfolio_stats

        except Exception as e:
            self.Error(f"Error in CalculatePortfolioRisk: {e}")
            return {}

    def CalculateValueAtRisk(self, returns, confidence=None):
        """Calcola il Value at Risk del portafoglio"""
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

    def CalculateSectorBetas(self):
        """Calcola i beta per ogni settore nel portafoglio"""
        try:
            spy_history = self.History(self.Symbol("SPY"), 252, Resolution.Daily)
            spy_returns = spy_history['close'].pct_change().dropna()
            
            for sector in set(self.sector_exposures.keys()):
                sector_symbols = [s for s in self.Portfolio.Keys 
                                if self.Securities[s].Fundamentals.AssetClassification.MorningstarSectorCode == sector]
                
                if sector_symbols:
                    sector_returns = pd.DataFrame()
                    for symbol in sector_symbols:
                        history = self.History([symbol], 252, Resolution.Daily)
                        if not history.empty:
                            sector_returns[str(symbol)] = history['close'].pct_change()
                    
                    sector_portfolio_returns = sector_returns.mean(axis=1)
                    beta = self.CalculateBeta(sector_portfolio_returns, spy_returns)
                    self.risk_metrics['sector_betas'][sector] = beta
                    
                    if beta > self.risk_params['max_sector_beta']:
                        self.Log(f"Warning: High beta ({beta:.2f}) detected in sector {sector}")

        except Exception as e:
            self.Error(f"Error in CalculateSectorBetas: {e}")

    def MonitorLiquidity(self):
        """Monitora la liquidità dei titoli in portafoglio"""
        try:
            for symbol in self.Portfolio.Keys:
                history = self.History([symbol], self.risk_params['min_liquidity_window'], Resolution.Daily)
                if not history.empty:
                    avg_volume = history['volume'].mean()
                    avg_price = history['close'].mean()
                    daily_liquidity = avg_volume * avg_price
                    
                    self.risk_metrics['liquidity_scores'][symbol] = daily_liquidity
                    
                    position_value = self.Portfolio[symbol].HoldingsValue
                    if position_value > daily_liquidity * 0.1:  # Non più del 10% del volume giornaliero
                        self.Log(f"Warning: Potential liquidity issue with {symbol}")

        except Exception as e:
            self.Error(f"Error in MonitorLiquidity: {e}")

    def CheckLiquidityConstraints(self, weights):
        """Verifica i vincoli di liquidità per i pesi proposti"""
        try:
            liquidity_scores = []
            for symbol, weight in zip(self.selectedStocks, weights):
                if symbol in self.risk_metrics['liquidity_scores']:
                    position_value = self.Portfolio.TotalPortfolioValue * weight
                    liquidity_score = position_value / self.risk_metrics['liquidity_scores'][symbol]
                    liquidity_scores.append(liquidity_score)
            
            return -max(liquidity_scores) + 0.1  # Constraint: position_value <= 10% of daily liquidity

        except Exception as e:
            self.Error(f"Error in CheckLiquidityConstraints: {e}")
            return 0

    # ...altri metodi se necessari...
