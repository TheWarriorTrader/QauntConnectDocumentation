<?php echo "<p>The following table describes the supported <code>Market</code> values:</p>"; ?>

<?php echo file_get_contents(DOCS_RESOURCES."/enumerations/market.html"); ?>

<?php echo "<p>LEAN groups all of the US Equity exchanges under <code>Market.USA</code>. In live mode, the brokerage routes the orders to the exchange that provides the best price.</p>"; ?>
