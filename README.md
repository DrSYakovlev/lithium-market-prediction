![Lithium symbol](./assets/img/lithium_symbol_title_image.jpg)
[Source](https://investingnews.com/daily/resource-investing/battery-metals-investing/lithium-investing/lithium-production-by-country/)

# Lithium Weekly Forecast

INTRODUCTION

This project analyzes the behavior of the lithium market by predicting whether the stock prices of major lithium producers will move up or down in the following week. It combines financial time series data with significant industry events to enhance prediction accuracy and technical analysis indicators. The model helps to identify market trends using machine learning. Visual insights are provided through an interactive dashboard. The aim is to support better decision-making in the lithium investment space.

DISCLAIMER

The information and predictions presented in this project are for educational and informational purposes only. They do not constitute investment advice, financial guidance, or a recommendation to buy or sell any securities. While efforts have been made to ensure the accuracy of the data and models used, no guarantee is provided regarding the reliability, completeness, or future performance of any predictions. Always conduct your own research and consult with a licensed financial advisor before making any investment decisions. The creators of this project are not responsible for any financial losses incurred as a result of using or relying on the content presented.

---

## Table of contents
### [I. Project goal](#i-project-goal-1)
### [II. Data Strategy](#ii-data-strategy-1)
### [III. ML pipeline](#iii-ml-pipeline-1)
#### [Live project](#live-project-1)


---
#### I. Project goal
Train **XGBoost** model to be able to predict the next-week lithium market sentiment (Bullish / Bearish).
#### II. Data strategy
* Historical and the most recent market data (<em>weekly interval </em>) are coming from **yfinance** open-source library
* Library of influential events, relevant to Li-ion battery sector is generated using OpenAI ChatGPT
* Feature Engineering was performed using Technical Analysis Python library; additional features such as volatility, momentum or moving averages were added
* The data were merged for model training and prediction purpose
* **Time-based** train-test data splitting enabled testing a trained model on previously unseen "future" data (relative to the training dataset); in this case we can expect the model does not suffer from overfitting

#### <u>Proxy target</u>
A lithium index was calculated based on the share price of three major Li producers:


| Ticker | Company Name                  | Country | Description                                                                                                     |
|--------|-------------------------------|---------|-----------------------------------------------------------------------------------------------------------------|
| ALB    | Albemarle Corporation         | USA     | One of the world’s largest lithium producers. Supplies lithium for batteries (EVs, storage, etc.). Also involved in bromine and catalysts. |
| SQM    | Sociedad Química y Minera de Chile | Chile   | Major Chilean chemical company. Big player in lithium extraction from brine in the Atacama Desert.             |
| PLL    | Piedmont Lithium Inc.         | USA     | Lithium exploration and development company with U.S. operations and supply deals with EV makers.              |

Sentiment of this index (Bullish or Bearish) at week closing time of the market was taken as a proxy target for prediction.

#### III. ML pipeline
ML pipeline major steps:
1. Data collection
2. Feature engineering
3. Model building
4. Model evaluation

#### Live project
[Live dashboard on Render](https://lithium-market-prediction.onrender.com/app)

---

[Author](https://www.linkedin.com/in/sergey-yakovlev-823514295/)

