import yfinance as yf
import pandas as pd
import panel as pn
import hvplot.pandas  # needed for plotting
from sklearn.metrics import classification_report, confusion_matrix


def events():
    """
    Returns library of relevant hystorical events
    Note: yyyy-mm-dd is the standard yfinance data format
    """
    li_ion_events = {
        "1976-01-01": "John B. Goodenough proposes cobalt oxide as a cathode material.",
        "1980-01-01": "Introduction of the layered cathode structure for lithium-ion batteries by John B. Goodenough.",
        "1985-01-01": "Akira Yoshino develops the first prototype of a lithium-ion battery.",
        "1991-01-01": "Sony commercializes the first lithium-ion rechargeable battery.",
        "2008-01-01": "Tesla launches the Roadster, the first EV using Li-ion batteries.",
        "2010-12-01": "Nissan Leaf becomes the world's first mass-produced EV with Li-ion batteries.",
        "2012-06-01": "Tesla Model S launches, popularizing electric vehicles with long-range capability.",
        "2016-07-01": "Tesla's Gigafactory 1 in Nevada begins operation.",
        "2017-01-01": "China launches significant subsidies for EVs, increasing battery production facilities.",
        "2018-01-01": "LG Chem announces major battery plant in Poland.",
        "2019-10-09": "Nobel Prize awarded to Goodenough, Whittingham, and Yoshino for their work on Li-ion batteries.",
        "2019-01-01": "Hyundai Kona Electric and Tesla Model 3 redefine affordable long-range EVs.",
        "2020-01-01": "Tesla's Gigafactory Shanghai becomes fully operational.",
        "2020-03-01": "COVID-19 pandemic disrupts global supply chains.",
        "2022-01-01": "Solid-state batteries reach early commercial prototypes.",
        "2022-08-01": "Lithium prices skyrocket due to supply-demand imbalances.",
        "2023-01-01": "U.S. Inflation Reduction Act provides subsidies for EVs and domestic battery manufacturing.",
        "2023-01-01": "Major advancements in lithium recycling technology.",
        "2023-06-01": "CATL announces new energy-dense battery for EVs.",
        "2024-01-01": "EU announces strict EV regulations, banning internal combustion engine sales by 2035.",
        "2025-01-01": "Tesla's Gigafactory Berlin expected to reach peak production capacity."
    }
    return li_ion_events


def disclaimer():
    """
    returns disclaimer Panel pane
    """
    disclaimer = pn.pane.Markdown(
        f"""
        ##### <u>Disclaimer</u>: The information and predictions presented in this project are for educational and informational purposes only. They do not constitute investment advice, financial guidance, or a recommendation to buy or sell any securities. While efforts have been made to ensure the accuracy of the data and models used, no guarantee is provided regarding the reliability, completeness, or future performance of any predictions. Always conduct your own research and consult with a licensed financial advisor before making any investment decisions. The creators of this project are not responsible for any financial losses incurred as a result of using or relying on the content presented.""",
        sizing_mode="stretch_width")
    return disclaimer


def get_weekly_data(tickers, start, end):
    """
    Load market data to dataframe
    """
    df = yf.download(tickers, start=start, end=end, interval='1wk')['Close']
    df = df.dropna(how='all')  # drop rows with all NaNs
    return df


def create_page_1(color, next_week, label_str, confidence_str, disclaimer):
    """
    Shows current (live) lithium index sentiment.
    """
    hr = pn.pane.Markdown("<hr>", sizing_mode="stretch_width")
    forecast_card = pn.pane.Markdown(
        f"""
        <div style="color:{color}; font-size:18px;">
        🔮 Forecast for the week of: 📅 {next_week.strftime('%Y-%m-%d')}<br>
        📊 Sentiment: <strong>{label_str}</strong><br>
        🎯 Confidence (Bullish): {confidence_str}
        </div>
        """,
        sizing_mode="stretch_width"
    )

    return pn.Column(
        pn.pane.Markdown("## 🔋Next week predicted Li sentiment"),
        forecast_card,
        hr,
        disclaimer,
        align="center"
    )


def create_page_2():
    """
    shows information about the project
    """
    info_card = pn.pane.Markdown(
        f"""
        - The **App** predicts if the next week lithium
        index market sentiment is **bullish** or **bearish**
        - Prediction is made using trained XGBoost classifier
        - Training dataset combines:
            - Historical market data from
            [yfinance](https://pypi.org/project/yfinance/)
            (calculated lithium index)
            - Features engineered using technical analysis
            - Library of influential historical events (Li-battery-specific)
        """,
        sizing_mode="stretch_width")

    return pn.Column(
        pn.pane.Markdown("## 📖 About the project", sizing_mode="stretch_width"),
        pn.pane.Markdown(
            "MVP - dashboard showing model applicability",
            sizing_mode="stretch_width"),
        pn.pane.Markdown(
            "[Project repo](https://github.com/DrSYakovlev/lithium-market-prediction)"),
        info_card,
        disclaimer,
        align="center")


def create_page_3(returns_df, model_df, disclaimer):
    """
    Shows time series graph of lithium index and model predictions.

    Parameters:
    - returns_df: DataFrame with historical lithium market data
    - model_df: DataFrame with predictions and confidence
    - disclaimer: Panel component (e.g., Markdown) with a disclaimer
    """
    date_slider = pn.widgets.DateRangeSlider(
        name='Date Range',
        start=returns_df.index.min(),
        end=returns_df.index.max(),
        value=(returns_df.index.min(), returns_df.index.max())
    )

    @pn.depends(date_slider.param.value)
    def sentiment_plot(date_range):
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        df = model_df[(model_df["Date"] >= start) & (model_df["Date"] <= end)]

        plot = df.hvplot.line(
            x="Date", y="Lithium_Market_Index",
            label="Lithium Index", line_width=2, color="dodgerblue")

        bull = df[df["Sentiment_Label"] == 1].hvplot.scatter(
            x="Date", y="Lithium_Market_Index", color="green", size=40, marker="^", label="Bullish")

        bear = df[df["Sentiment_Label"] == 0].hvplot.scatter(
            x="Date", y="Lithium_Market_Index", color="red", size=40, marker="v", label="Bearish")

        pred_bull = df[df["Predicted_Label"] == 1].hvplot.scatter(
            x="Date", y="Lithium_Market_Index", color="blue", marker="*", size=100, label="Predicted Bullish")

        incorrect = df[(df["Predicted_Label"] == 1) & (df["Correct"] == False)].hvplot.scatter(
            x="Date", y="Lithium_Market_Index", color="yellow", marker="x", size=100, label="Incorrect Prediction")

        return plot * bull * bear * pred_bull * incorrect

    return pn.Column(
        pn.pane.Markdown("## 🔍 Explore sentiment predictions by the model"),
        date_slider,
        sentiment_plot,
        pn.pane.Markdown('<hr>', sizing_mode="stretch_width"),
        disclaimer,
        align="center"
    )


def create_page_4(y_test, y_pred, disclaimer):
    """
    Shows model performance metrics and confusion matrix.

    Parameters:
    - y_test: array-like – true sentiment labels
    - y_pred: array-like – predicted sentiment labels
    - disclaimer: Panel component with footer/disclaimer
    """
    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_long = pd.DataFrame(
        conf_mat, index=["Actual Bearish", "Actual Bullish"],
        columns=["Predicted Bearish", "Predicted Bullish"]
    ).reset_index().melt(
        id_vars="index", var_name="Predicted", value_name="Count"
    ).rename(columns={"index": "Actual"})

    cm_plot = cm_long.hvplot.heatmap(
        x='Predicted',
        y='Actual',
        C='Count',
        cmap='Blues',
        line_color='white',
        colorbar=True,
        height=300, width=450
    )

    # Precision / Recall / F1-score
    report_dict = classification_report(
        y_test, y_pred, target_names=["Bear", "Bull"], output_dict=True)

    report_df = pd.DataFrame(report_dict).T.iloc[:2]  # Only Bear & Bull
    report_df['Class'] = report_df.index
    report_long = report_df.melt(
        id_vars="Class",
        value_vars=["precision", "recall", "f1-score"],
        var_name="Metric",
        value_name="Score"
    )

    report_bar = report_long.hvplot.bar(
        x="Metric",
        y="Score",
        by="Class",
        rot=0,
        ylabel="Score",
        xlabel="Metric",
        height=350,
        width=600,
        legend='top_right',
        color=['#1f77b4', '#ff7f0e']
    )

    return pn.Column(
        pn.pane.Markdown("## 📋 Model performance"),
        pn.pane.Markdown("### 🧠 Confusion matrix (test set)"),
        cm_plot,
        pn.pane.Markdown('<hr>', sizing_mode="stretch_width"),
        pn.pane.Markdown("### 📊 Precision / Recall / F1-Score"),
        report_bar,
        pn.pane.Markdown('<hr>', sizing_mode="stretch_width"),
        disclaimer,
        align="center"
    )
