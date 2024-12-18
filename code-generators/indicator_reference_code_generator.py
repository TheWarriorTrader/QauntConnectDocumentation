import base64
import json
import numpy as np
import os
import pathlib
from pathlib import Path
import re
from shutil import rmtree
from urllib.request import urlopen
from json import dumps
from _code_generation_helpers import get_type, to_key, generate_landing_page, title_to_dash_linked_lower_case, INDICATORS

TAG = f'<!-- Code generated by {os.path.basename(__file__)} -->'
OPTION_INDICATORS = ["ImpliedVolatility", "Delta", "Gamma", "Vega", "Theta", "Rho"]
EXCEPTION_CS_FILES = {"Stochastic.cs": "Stochastics.cs"}

def _format_introduction(type_name: str, text: str) -> str:
    if 'CandlestickPatterns' in type_name:
        return f"Create a new {text} to indicate the pattern's presence."

    text = text.replace("Represents", "This indicator represents")
    if "Source: " in text:
        link_split = text.split("http")
        return link_split[0].replace("Source: ", f'<sup><a href="https{link_split[1]}">source</a></sup>'.replace("httpss", "https"))
    return text

PROPERTIES_EXCEPTIONS = ['MovingAverageType', 'IsReady', 'WarmUpPeriod', 'Name', 'Period', 'Samples', 
                'Current', "Consolidators", "Previous", "Window", "[System.Int32]", "Strike", "Right", 
                "Style", "Expiry", "UseMirrorContract", "GetEnumerator", "moving_average_type", "is_ready", "warmup_period",
                "name", "period", "samples", "current", "consolidators", "previous", "window", "int", "strike",
                "right", "style", "expiry", "use_mirror_contract", "warm_up_period", "item", "get_enumerator"]

def _extract_properties(properties: list):
    numerical_properties = ''
    indicator_properties = ''
    for property in properties:
        property_name = property["property-name"]
        if property_name in PROPERTIES_EXCEPTIONS:
            continue
        # Some properties are lists we cannot plot
        full_type = property['property-full-type-name'] 
        if full_type.startswith('System.Collection'):
            continue
        if full_type.startswith('QuantConnect'):
            indicator_properties += f'"{property_name}",'
        else:
            numerical_properties += f'"{property_name}",'

    return f'array({indicator_properties[:-1]})', f'array({numerical_properties[:-1]})'

def _get_helpers():
    with open(f'Resources/indicators/IndicatorImageGenerator.py', mode='r') as fp:
        lines = fp.readlines()
        helpers = {}
        for i, line in enumerate(lines):
            if 'title' in line and ':' in line:
                name = lines[i-3].split(':')[0].strip()[1:-1]
                full_constructor = lines[i-1]
                parts = line.split('(')

                helpers[to_key(name)] = {
                    'method': parts[0].split(' ')[-1][1:], 
                    'arguments': ')'.join('('.join(parts[1:]).split(')')[:-1]),
                    'constructor-arguments': ')'.join('('.join(full_constructor.split('(')[1:]).split(')')[:-1])
                }

        return helpers

def _get_image_source(folder: str) -> str:
    image = '/'.join([part[3:].strip().lower().replace(' ','-') for part in folder.parts])
    return f'https://cdn.quantconnect.com/docs/i/{image}.png'

def Generate_Indicators_Reference():
    indicators = dict()
    helpers = _get_helpers()

    path = Path('Resources/indicators/constructors')
    for file in path.iterdir():
                
        with open(file, mode='r', encoding="utf-8") as fp:
            try:
                content = fp.read()
                start = content.find('QuantConnect')
                type_name = content[start: content.find('(', start)].split("</code>")[0].strip()
                
                indicator = get_type(type_name, "csharp")
                py_indicator = get_type(type_name, "python")
                key = " ".join(re.findall('[a-zA-Z][^A-Z]*', indicator['type-name']))
                indicator['description'] = _format_introduction(type_name, indicator.get('description'))
                type_name = indicator['type-name'] if not indicator['type-name'].startswith('Momersion') else 'Momersion'
                start = content.find('https://github.com/QuantConnect/Lean/blob/master/Indicators/')   
                indicator_source = f"https://github.com/QuantConnect/Lean/tree/master/Indicators/{type_name}.cs" \
                    if not "CandlestickPatterns" in indicator["full-type-name"] else \
                    f"https://github.com/QuantConnect/Lean/tree/master/Indicators/CandlestickPatterns/{indicator['type-name']}.cs"
                for wrong_file, correct_file in EXCEPTION_CS_FILES.items():
                    indicator_source = indicator_source.replace(wrong_file, correct_file)
                indicator['source'] = indicator_source
                
                helper = helpers.get(file.stem, {
                    'method': indicator['type-name'], 
                    'arguments': "symbol",
                    'constructor-arguments': None
                } )

                arguments = helper['arguments']
                indicator['helper-name'] = helper['method'] 
                indicator['helper-arguments'] = arguments
                start = arguments.find(',')
                if start > 0:
                    arguments = arguments[1 + start:].strip()
                indicator['constructor-arguments'] = helper['constructor-arguments']
                indicator['has-moving-average-type-parameter'] = 'MovingAverageType' in content
                indicator['properties'] = _extract_properties(indicator['properties'])
                indicator['py_properties'] = _extract_properties(py_indicator['properties'])

                indicators[key] = indicator
            
            except Exception as e:
                print(f"Cannot generate {file.stem} - {e}")
                continue

    types = {
        'Indicator': {
            'name': 'data-point-indicator',
            'update-parameter-type': 'time/number pair or an <code>IndicatorDataPoint</code>',
            'update-parameter-value': 'bar.EndTime, bar.Close'
        },
        'BarIndicator': {
            'name': 'bar-indicator',
            'update-parameter-type': 'a <code>TradeBar</code> or <code>QuoteBar</code>',
            'update-parameter-value': 'bar'
        },
        'TradeBarIndicator': {
            'name': 'trade-bar-indicator',
            'update-parameter-type': 'a <code>TradeBar</code>',
            'update-parameter-value': 'bar'
        }
    }

    # Get Indicator Type
    def find_indicator_type(base_type):
        if 'CandlestickPatterns' in base_type:
            return types["TradeBarIndicator"]

        for k, v in types.items():
            if f'QuantConnect.Indicators.{k}' in base_type:
                return v
        key = ' '.join(re.findall('[a-zA-Z][^A-Z]*', base_type.split('.')[-1]))
        base = indicators.get(key, get_type(base_type, "csharp"))
        return find_indicator_type(base['base-type-full-name'])

    for key, indicator in indicators.items():
        indicator_type = find_indicator_type(indicator['base-type-full-name'])
        indicator['update-parameter-type'] = indicator_type['update-parameter-type']
        indicator['update-parameter-value'] = indicator_type['update-parameter-value']

    # DELETE ALL FILES
    rmtree(INDICATORS, ignore_errors=True)
    Path(f'{INDICATORS}/00 Candlestick Patterns/').mkdir(parents=True, exist_ok=True)

    count = 0
    candle = 0
    for key in sorted(indicators.keys()):
        indicator = indicators.get(key)
        if 'CandlestickPatterns' in indicator['full-type-name']:
            candle += 1
            indicator['folder'] = Path(f'{INDICATORS}/00 Candlestick Patterns/{candle:02} {key}')
        else:
            count += 1
            indicator['folder'] = Path(f'{INDICATORS}/{count:03} {key}')

    with open('Resources/indicators/indicator_count.html', 'w', encoding='utf-8') as fp:
        fp.write(f'There are {count} indicators.')

    with open('Resources/indicators/candlestick_pattern_count.html', 'w', encoding='utf-8') as fp:
        fp.write(f'There are {candle} candlestick pattern indicators.')

    generate_landing_page(0, count, INDICATORS, 'Supported Indicators',
        '<p>Indicators translate a stream of data points into a numerical value you can use to detect trading opportunities. LEAN provides more than 100 pre-built technical indicators and candlestick patterns you can use in your algorithms. You can use any of the following indicators. Click one to learn more.</p>')

    generate_landing_page(1, 1+candle, f'{INDICATORS}/00 Candlestick Patterns', 'Candlestick Patterns',
        '<p>You can use any of the following candlestick patterns. Click one to learn more.</p>')

    for key, indicator in indicators.items():
        folder = indicator['folder']
        folder.mkdir(parents=True, exist_ok=True)

        type_name = indicator['type-name']
        description = indicator['description']
        helper_name = indicator['helper-name']
        py_helper_name = title_to_dash_linked_lower_case(helper_name).replace("-", "_")
        image_source = _get_image_source(folder)
        source = indicator['source']

        with open(f'{folder}/01 Introduction.html', 'w', encoding='utf-8') as fp:
            category = 'candlestick pattern' if 'CandlestickPatterns' in source else 'indicator'
            fp.write(f"""{TAG}
<p>{description}</p>
<p>To view the implementation of this {category}, see the <a rel="nofollow" target="_blank" href="{source}">LEAN GitHub repository</a>.</p>""")

        with open(f'{folder}/02 Using {helper_name} Indicator.php', 'w', encoding='utf-8') as fp:
            fp.write(f"""{TAG}
<? 
$hasReference = { 'true' if 'reference' in indicator['helper-arguments'] else 'false' };
$hasAutomaticIndicatorHelper = {'true' if type_name != 'Delay' else 'false'};
$helperPrefix = '{'CandlestickPatterns.' if 'CandlestickPatterns' in source else ''}';
$typeName = '{type_name}';
$helperName = '{helper_name}';
$pyHelperName = '{py_helper_name}';
$helperArguments = '{indicator['helper-arguments']}';
$properties = {indicator['properties'][0]};
$pyProperties = {indicator['py_properties'][0]};
$otherProperties = {indicator['properties'][1]};
$otherPyProperties = {indicator['py_properties'][1]};
$updateParameterType = '{indicator['update-parameter-type']}';
$constructorArguments = '{indicator['constructor-arguments'] if indicator['constructor-arguments'] else ''}';
$updateParameterValue = '{indicator['update-parameter-value']}';
$hasMovingAverageTypeParameter = {indicator['has-moving-average-type-parameter']};
$constructorBox = '{key.lower().replace(' ','-')}';
$isOptionIndicator = { 'true' if type_name in OPTION_INDICATORS else 'false' };
include(DOCS_RESOURCES."/indicators/using-indicator.php");
?>""")

        if 'CandlestickPatterns' not in indicator['full-type-name']:
            with open(f'{folder}/03 Visualization.php', 'w', encoding='utf-8') as fp:
                fp.write(f"""{TAG}
<?
$typeName = "{type_name}";
$imageSource = "{image_source}";
include(DOCS_RESOURCES."/indicators/visualization.php");
?>""")

        if description.find('<see cref=\"T:') > 0:
            description = description.replace('<see cref=\"T:','').replace('\" />','')
        description = re.sub(r'<a href=".*?">(.*?)</a>', r'\1', description)
        description = re.sub(r'<sup>(.*?)</sup>', "", description)
        if len(description) > 127:
            description = description[:127] + '...'
        with open(f'{folder}/metadata.json', 'w', encoding='utf-8') as fp:
            metadata = {
                'type': 'metadata',
                'values': {
                    'description': description,
                    'keywords': key.lower(),
                    'og:type': 'website',
                    'og:description': description,
                    'og:title': f'{key} - Using Indicators on QuantConnect.com',
                    'og:site_name': f'{key} - Using Indicators on QuantConnect.com',
                    'og:image': image_source
                }
            }
            fp.write(dumps(metadata, indent=4))

        with open(f'{folder}/04 Indicator History.php', 'w', encoding='utf-8') as fp:
            fp.write(f"""{TAG}
<? 
$helperPrefix = '{'CandlestickPatterns.' if 'CandlestickPatterns' in source else ''}';
$typeName = '{type_name}';
$helperName = '{helper_name}';
$pyHelperName = '{py_helper_name}';
$helperArguments = '{indicator['helper-arguments']}';
$hasReference = { 'true' if 'reference' in indicator['helper-arguments'] else 'false' };
$isOptionIndicator = { 'true' if type_name in OPTION_INDICATORS else 'false' };
$csharpProperties = {indicator['properties'][0]};
$pythonProperties = {indicator['py_properties'][0]};
include(DOCS_RESOURCES."/indicators/indicator-history.php");
?>""")
            
if __name__ == "__main__":
    Generate_Indicators_Reference()