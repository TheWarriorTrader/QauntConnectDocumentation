<p>The <code>Reader</code> method of your custom data class takes one line of data from the source location and parses it into one of your custom objects. You can add as many properties to your custom data objects as you need, but the following table describes the properties you must set. When there is no useable data in a line, the method should return <code class="csharp">null</code><code class="python">None</code>.</p>

<table class="qc-table table">
    <thead>
        <tr>
            <th>Property</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>Symbol</code></td>
            <td>Set this property to <code>config.Symbol.</code></td>
        </tr>
        <tr>
            <td><code>EndTime</code></td>
            <td>The time when the data sample ends and when LEAN should add the sample to a <a href="/docs/v2/writing-algorithms/key-concepts/time-modeling/timeslices">Slice</a>.</td>
        </tr>
        <tr>
            <td><code>Value</code></td>
            <td>The default data point value.<br></td>
        </tr>
    </tbody>
</table>



<div class="section-example-container">
<pre class="csharp">public class MyCustomDataType : BaseData
{
    public decimal Property1 = 0;
    public string errString = "";

    public override BaseData Reader(
        SubscriptionDataConfig config,
        string line,
        DateTime date,
        bool isLive)
    {
        var data = line.Split(',');
        return new MyCustomDataType()
        {
            // Make sure we only get this data AFTER trading day - don't want look-ahead bias.
            EndTime = DateTime.ParseExact(data[0], "yyyyMMdd", null).AddHours(20),
            Symbol = config.Symbol,
            Value = Convert.ToDecimal(data[1]),
            Property1 = Convert.ToDecimal(data[2])
        };
    }
}
</pre>
<pre class="python">class MyCustomDataType(PythonData):
    def Reader(self, config: SubscriptionDataConfig, line: str, date: datetime, isLive: bool) -&gt; BaseData:
        data = line.split(',')
        custom = MyCustomDataType()
        custom.Symbol = config.Symbol

        # Make sure we only get this data AFTER trading day - don't want look-ahead bias.
        custom.EndTime = datetime.strptime(data[0], '%Y%m%d') + timedelta(hours=20) 

        custom.Value = float(data[1])
        custom["property1"] = float(data[2])
        return custom
</pre>
</div>

<?php echo file_get_contents(DOCS_RESOURCES."/datasets/custom-data/reader-method.html"); ?>

<div class="section-example-container">
<pre class="python">class MyCustomDataType(PythonData):
    def GetSource(self, config: SubscriptionDataConfig, date: datetime, isLiveMode: bool) -&gt; SubscriptionDataSource:
        return SubscriptionDataSource("https://raw.githubusercontent.com/DerekMelchin/custom-data-test-2/main/abcd.json", SubscriptionTransportMedium.RemoteFile, FileFormat.UnfoldingCollection)

    def Reader(self, config: SubscriptionDataConfig, line: str, date: datetime, isLiveMode: bool) -&gt; BaseData:
        objects = []
        data = json.loads(line)
        endTime = None
        for j, i in enumerate(data):
            coin = Bitcoin()
            coin.Symbol = config.Symbol

            ts = int(i['timestamp'])
            coin.Time = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            coin.EndTime = coin.Time + timedelta(days=1)
            endTime = coin.EndTime

            coin.Value = j

            objects.append(coin)
        return BaseDataCollection(endTime, config.Symbol, objects)</pre>
</div>&nbsp;^ Example: https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_444c6a94f8dd6f6b538ff6e5466aa0c7.html

<br>
