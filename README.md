![QuantConnect Logo](https://cdn.quantconnect.com/web/i/logo-small.png)
# QuantConnect Documentation

[Documentation][1] | [LEAN API][2] | [Download LEAN][3] | [Discord Channel][4]
----------

This repository is the primary source for documentation for QuantConnect.com. If you have edits to the documentation, please submit a pull request for the fix.

Contributors of accepted pull requests will receive a 2 month free Prime subscription on QuantConnect. Once your pull request has been merged, write to us at [support@quantconnect.com][5] with a link to your PR to claim your free live trading! =) QuantConnect <3 open source.

Lean Engine is an open source fully-managed C# algorithmic trading engine built for desktop and cloud usage. It was designed in Mono and operates on Windows, Linux, and Mac platforms. For more information about the LEAN Algorithmic Trading engine, see the [Lean Engine Repository][6].

## Table of Contents
1. [Objective of Structuring the Strategy](#objective-of-structuring-the-strategy)
2. [Overview of Machine Learning Support](#overview-of-machine-learning-support)
3. [Key Components of the Strategy](#key-components-of-the-strategy)
4. [Usage Instructions](#usage-instructions)
5. [New Documentation Requests and Edits](#new-documentation-requests-and-edits)
6. [Contributors and Pull Requests](#contributors-and-pull-requests)

## Objective of Structuring the Strategy

The objective is to structure the strategy as described in `strategia_spiegata.txt` and `knowledge_base.md`. This involves understanding the macroeconomic context, selecting sectors that align with that context, and choosing individual stocks based on fundamental analysis. Machine learning is used as a support tool to refine estimates and reduce the risk of overfitting. The entire approach is developed in Python and integrated with QuantConnect.

## Overview of Machine Learning Support

Machine learning is used in the strategy to support and refine the estimates made during the analysis. It helps in reducing the risk of overfitting by providing more accurate predictions based on historical data. The machine learning models are not the primary basis of the strategy but serve as a tool to enhance the decision-making process.

## Key Components of the Strategy

1. **Data Sources**: The strategy uses macroeconomic data from the FRED database integrated with QuantConnect. It also utilizes fundamental data for individual stocks.
2. **Analysis Methods**: The strategy involves a comprehensive analysis of the macroeconomic context, sector selection, and fundamental analysis of individual stocks.
3. **Optimization Techniques**: The strategy employs optimization techniques to create a balanced and resilient portfolio. This includes considering the dynamics between selected stocks and their potential reactions to macroeconomic changes.

## Usage Instructions

To use this repository, follow these steps:

1. **Set Up the Environment**: Ensure you have Python and QuantConnect installed on your system.
2. **Run the Code**: Navigate to the `src` directory and run the `main.py` file to execute the strategy.
3. **Access the Documentation**: Refer to the `docs` directory for detailed explanations and examples of the strategy implementation.

## New Documentation Requests and Edits

Please submit new documentation requests as an issue to the [Documentation repository][7]. Before submitting an issue, please read others to ensure it is not a duplicate. Edits and fixes for clarity are warmly welcomed!

## Contributors and Pull Requests

Contributions are very warmly welcomed, but we ask you read the existing code to see how it is formatted and commented, and to ensure contributions match the existing style. All code submissions must include accompanying tests. Please see the [contributor guidelines][8].

[1]: https://www.quantconnect.com/docs
[2]: https://www.quantconnect.com/docs/v2/lean-cli/api-reference
[3]: https://github.com/QuantConnect/Lean/archive/master.zip
[4]: https://www.quantconnect.com/slack
[5]: mailto:support@quantconnect.com
[6]: https://github.com/QuantConnect/Lean
[7]: https://github.com/QuantConnect/Documentation/issues
[8]: https://github.com/QuantConnect/Lean/blob/master/CONTRIBUTING.md
