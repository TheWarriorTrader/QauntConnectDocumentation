<!-- Code generated by indicator_reference_code_generator.py -->
<? 
$hasReference = false;
$hasAutomaticIndicatorHelper = true;
$helperPrefix = '';
$typeName = 'Delta';
$helperName = 'D';
$pyHelperName = 'd';
$helperArguments = 'option_symbol, option_mirror_symbol';
$properties = array("ImpliedVolatility","OptionSymbol","RiskFreeRate","DividendYield","Price","OppositePrice","UnderlyingPrice");
$pyProperties = array("implied_volatility","option_symbol","risk_free_rate","dividend_yield","price","opposite_price","underlying_price");
$otherProperties = array();
$otherPyProperties = array();
$updateParameterType = 'time/number pair or an <code>IndicatorDataPoint</code>';
$constructorArguments = 'option_symbol, interest_rate_model, dividend_yield_model, option_mirror_symbol';
$updateParameterValue = 'bar.EndTime, bar.Close';
$hasMovingAverageTypeParameter = False;
$constructorBox = 'delta';
$isOptionIndicator = true;
include(DOCS_RESOURCES."/indicators/using-indicator.php");
?>