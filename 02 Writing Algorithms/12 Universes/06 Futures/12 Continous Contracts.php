<div data-tree='QuantConnect.DataNormalizationMode'></div>

<p>By default, LEAN only subscribes to the continuous Future contract. A continuous Future contract represents a series of separate contracts stitched together to form a continuous price. If you need a lot of historical data to warm up an indicator, apply the indicator to the continuous contract price series. The <code>Future</code> object has a <code>Symbol</code> property and a <code>Mapped</code> property. The price of the <code>Symbol</code> property is the adjusted price of the continuous contract. The price of the <code>Mapped</code> property is the raw price of the currently selected contract in the continuous contract series.</p>

<div class="section-example-container">
    <pre class="csharp">// Get the adjusted price of the continuous contract
var adjustedPrice = Securities[_future.Symbol].Price; 

// Get the raw price of the currently selected contract in the continuous contract series
var rawPrice = Securities[_future.Mapped].Price;</pre>
    <pre class="python"># Get the adjusted price of the continuous contract
adjusted_price = self.Securities[self.future.Symbol].Price 

# Get the raw price of the currently selected contract in the continuous contract series
raw_price = self.Securities[self.future.Mapped].Price</pre>
</div>


<p>To configure how LEAN identifies the current Future contract in the continuous series and how it forms the adjusted price between each contract, provide <code>dataMappingMode</code>, <code>dataNormalizationMode</code>, and <code>contractDepthOffset</code> arguments to the <code>AddFuture</code> method. The <code>Future</code> object that the <code>AddFuture</code> method returns contains a <code>Mapped</code> property that references the current contract in the continuous contract series. As the contracts roll over, the <code>Mapped</code> property references the next contract in the series and you receive a <code>SymbolChangedEvent</code> object in the <code>OnData</code> method. The <code>SymbolChangedEvent</code> references the old contract <code>Symbol</code> and the new contract <code>Symbol</code>.</p>

<div class="section-example-container">
    <pre class="csharp">public override void OnData(Slice slice)
{
    foreach (var changedEvent in slice.SymbolChangedEvents.Values)
    {
        Log($"Symbol changed: {changedEvent.OldSymbol} -> {changedEvent.NewSymbol}");
    }
}</pre>
    <pre class="python">def OnData(self, slice: Slice) -> None:
    for changed_event in slice.SymbolChangedEvents.Values:
        self.Log(f"Contract rollover from {changed_event.OldSymbol} to {changed_event.NewSymbol}")</pre>
</div>

<h4>Data Normalization Modes</h4>
<p>The <code>dataNormalizationMode</code> argument defines how the price series of two contracts are stitched together when the contract rollovers occur.</p>

<p>The following table describes the <code>DataNormalizatoinMode</code> enumerator members for continuous contracts:</p>

<table class="qc-table table">
<thead>
    <tr>
        <th style="width: 25%;">Member</th>
        <th style="width: 5%;">Value</th>
        <th style="width: 70%;">Description</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>Raw</td>
        <td>0</td>
        <td>No price adjustment between the two contracts.</td>
    </tr>
    <tr>
        <td>ForwardPanamaCanal</td>
        <td>4</td>
        <td>Eliminates price jumps between two consecutive contracts, adding a factor based on the difference of their prices. First contract is the true one, factor 0.</td>
    </tr>
    <tr>
        <td>BackwardsPanamaCanal</td>
        <td>5</td>
        <td>Eliminates price jumps between two consecutive contracts, adding a factor based on the difference of their prices. Last contract is the true one, factor 0.</td>
    </tr>
    <tr>
        <td>BackwardsRatio</td>
        <td>6</td>
        <td>Eliminates price jumps between two consecutive contracts, multiplying the prices by their ratio. Last contract is the true one, factor 1.</td>
    </tr>
</tbody>
</table>

<h4>Data Mapping Modes</h4>
<p>The <code>dataMappingMode</code> argument defines when contract rollovers occur. The <code>DataMappingMode</code> enumeration has the following members:</p>
<div data-tree="QuantConnect.DataMappingMode"></div>


<h4>Contract Depth Offsets</h4>
<p>The <code>contractDepthOffset</code> argument defines which contract to use. 0 (default) is the front month contract, 1 the following back month contract, and 3 is the second back month contract.</p>
