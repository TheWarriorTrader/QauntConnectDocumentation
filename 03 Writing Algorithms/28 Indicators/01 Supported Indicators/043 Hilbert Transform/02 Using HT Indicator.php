<!-- Code generated by indicator_reference_code_generator.py -->
<? 
$hasReference = false;
$hasAutomaticIndicatorHelper = true;
$helperPrefix = '';
$typeName = 'HilbertTransform';
$helperName = 'HT';
$pyHelperName = 'ht';
$helperArguments = 'symbol, 7, 0.635, 0.338';
$properties = array("InPhase","Quadrature");
$pyProperties = array("in_phase","quadrature");
$otherProperties = array();
$otherPyProperties = array();
$updateParameterType = 'time/number pair or an <code>IndicatorDataPoint</code>';
$constructorArguments = '7, 0.635, 0.338';
$updateParameterValue = 'bar.EndTime, bar.Close';
$hasMovingAverageTypeParameter = False;
$constructorBox = 'hilbert-transform';
$isOptionIndicator = false;
include(DOCS_RESOURCES."/indicators/using-indicator.php");
?>