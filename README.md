# Constructing time-series momentum portfolios with deep multi-task learning

## Abstract 
A diversified risk-adjusted time-series momentum (TSMOM) portfolio can deliver substantial abnormal returns and offer some degree of tail risk protection during extreme market events. The performance of existing TSMOM strategies, however, relies not only on the quality of the momentum signal but also on the efficacy of the volatility estimator. Yet many of the existing studies have always considered these two factors to be independent. Inspired by recent progress in Multi-Task Learning (MTL), we present a new approach using MTL in a deep neural network architecture that jointly learns portfolio construction and various auxiliary tasks related to volatility, such as forecasting realized volatility as measured by different volatility estimators. Through backtesting from January 2000 to December 2020 on a diversified portfolio of continuous futures contracts, we demonstrate that even after accounting for transaction costs of up to 3 basis points, our approach outperforms existing TSMOM strategies. Moreover, experiments confirm that adding auxiliary tasks indeed boosts the portfolio’s performance. These findings demonstrate that MTL can be a powerful tool in finance.

## Note
This repository contains code snippets that were used in the paper to produce the results. We believe this is more than sufficient to reproduce the results and serve as a guidance for readers who wish to extend the research.

- `feature.py`
- `metric.py`
- `model.py`

In addition, the bulk of the codes were not open-sourced as they are proprietary research framework code based. These include the backtesting framework, futures contracts price adjustments, etc. Having said that, this would not hamper your ability to conduct research, as the three scripts released above are more than sufficient to aid you.


Due to the data licensing agreement, we cannot share the data. There are a few ways to do this (i) you may use a Bloomberg Terminal to extract the futures contract's price and use a backward ratio adjusted method to stitch the prices together to obtain a continuous price series (ii) purchase the futures contracts pricing data from data vendors (i.e., CQG, Pinnacle Data, etc.). We are not affiliated with the data vendors and do not receive anything from mentioning them here. They are viable options for obtaining futures pricing data to conduct research.

For questions or suggestion, please write an email to joel_ong at mymail dot sutd dot edu dot sg
