<!-- Code generated by indicator_reference_code_generator.py -->
<? 
$hasReference = false;
$hasAutomaticIndicatorHelper = true;
$helperPrefix = '';
$typeName = 'SimpleMovingAverage';
$helperName = 'SMA';
$pyHelperName = 'sma';
$helperArguments = 'symbol, 20';
$properties = array("RollingSum");
$pyProperties = array("rolling_sum");
$otherProperties = array();
$otherPyProperties = array();
$updateParameterType = 'time/number pair or an <code>IndicatorDataPoint</code>';
$constructorArguments = '20';
$updateParameterValue = 'bar.EndTime, bar.Close';
$hasMovingAverageTypeParameter = False;
$constructorBox = 'simple-moving-average';
$isOptionIndicator = false;
include(DOCS_RESOURCES."/indicators/using-indicator.php");
?>