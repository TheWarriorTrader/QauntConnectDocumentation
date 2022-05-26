<div>- RegisterIndicator (save a reference to the consolidator so you can remove it later)</div><div>&nbsp;&nbsp;&nbsp;
 - See 
https://www.quantconnect.com/docs/v2/writing-algorithms/consolidating-data/updating-indicators#05-Remove-Indicator-Consolidators</div><div>&nbsp;&nbsp;&nbsp; - If you use RegisterIndicator, don't update the indicator in the consolidation handler (double updates)<br></div><div>&nbsp;&nbsp;&nbsp; - Can use any field you want <br></div><div><br></div>

<div class="section-example-container">
	<pre class="python"># request the equity data in minute resolution
self.AddEquity("SPY", Resolution.Minute)
# define a 10-period RSI indicator with indicator constructor
self.rsi = RelativeStrengthIndex(10, MovingAverageType.Simple)
# create the 30-minutes data consolidator
thirtyMinuteConsolidator = TradeBarConsolidator(timedelta(minutes=30))
self.SubscriptionManager.AddConsolidator("SPY", thirtyMinuteConsolidator)
# register the 30-minute consolidated bar data to automatically update the indicator
self.RegisterIndicator("SPY", self.rsi, thirtyMinuteConsolidator)
</pre>
	<pre class="csharp fsharp">// request the equity data in minute resolution
AddEquity(_symbol, Resolution.Hour);
// define a 10-period RSI indicator with indicator constructor
_rsi = new RelativeStrengthIndex(10, MovingAverageType.Simple);
// create the 30-minutes data consolidator
var thirtyMinuteConsolidator = new TradeBarConsolidator(TimeSpan.FromMinutes(30));
SubscriptionManager.AddConsolidator("SPY", thirtyMinuteConsolidator);
// register the 30-minute consolidated bar data to automatically update the indicator
RegisterIndicator("SPY", _rsi,thirtyMinuteConsolidator);
</pre>
</div>



<p>The data point indicators use only a single price data in their calculations. By default, those indicators use the closing price. For equity, that price is the trade bar closing price. For other asset classes with quote bar data (bid/ask price), those indicators are calculated with the mid-price of the bid closing price and the ask closing price.
</p>
<p>
	If you want to create an indicator with the other fields like <code>Open</code>, <code>High</code>, <code>Low</code>, or <code>Close</code>, you can specify the <code>selector</code> argument in the indicator helper method with the available fields.
</p>

<p>To register the price data with the specified field, call <code>RegisterIndicator</code>.</p>
<div class="section-example-container">
	<pre class="python"># define a 10-period RSI with indicator constructor
self.rsi = RelativeStrengthIndex(10, MovingAverageType.Simple)
# register the daily High price data to automatically update the indicator
self.RegisterIndicator("SPY", self.rsi, Resolution.Daily, Field.High)
</pre>
	<pre class="csharp">// define a 10-period RSI with indicator constructor
_rsi = new RelativeStrengthIndex(10, MovingAverageType.Simple);
// register the daily High price data to automatically update the indicator
RegisterIndicator("SPY", _rsi, Resolution.Daily, Field.High);
</pre>
</div>

<?php 
echo file_get_contents(DOCS_RESOURCES."/enumerations/field.html"); 
?>