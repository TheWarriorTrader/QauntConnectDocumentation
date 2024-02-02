<p>The <code>DataNormalizationMode</code> setting is an enumeration that defines how historical data is adjusted. This setting is only applicable for US Equities and Futures.

<p>In the case of US Equities, the data normalization mode affects how historical data is adjusted for <a href='/docs/v2/writing-algorithms/securities/asset-classes/us-equity/corporate-actions'>corporate actions</a>. To view all the available options, see <a href='/docs/v2/writing-algorithms/securities/asset-classes/us-equity/requesting-data#11-Data-Normalization'>Data Normalization</a>. To change the data normalization mode, in the <a href='/docs/v2/writing-algorithms/initialization'>Initialize</a> method, adjust the algorithm's <code>UniverseSettings</code> before you create the Universe Selection model.</p>

<p>In the case of Futures, the data normalization mode affects how historical data of two contracts is stitched together to form the <a href='/docs/v2/writing-algorithms/universes/futures#12-Continous-Contracts'>continuous contract</a>. To view all the available options, see <a href='/docs/v2/writing-algorithms/securities/asset-classes/futures/requesting-data#09-Data-Normalization'>Data Normalization</a>.</p>

<p>The default value is <code>DataNormalizationMode.Adjusted</code>. To change the data normalization mode, in the <a href='/docs/v2/writing-algorithms/initialization'>Initialize</a> method, adjust the algorithm's <code>UniverseSettings</code> before you create the Universe Selection model.</p>

<div class="section-example-container">
	<pre class="csharp">UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw;

var tickers = new[] {"MSTR", "MSFT", "IBM"};
var symbols = tickers.Select(ticker =&gt; QuantConnect.Symbol.Create(ticker, SecurityType.Equity, Market.USA));
AddUniverseSelection(new ManualUniverseSelectionModel(symbols));</pre>
	<pre class="python">self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw

tickers = ["MSTR", "MSFT", "IBM"]
symbols = [ Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in tickers]
self.AddUniverseSelection(ManualUniverseSelectionModel(symbols))</pre>
</div>