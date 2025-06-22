import glob
import os

import pandas as pd
import plotly.express as px

def plot_all_results():
    files = glob.glob("results/*.csv")
    if not files:
        print("Keine CSV-Dateien gefunden. Erst Training ausführen!")
        return

    all_df = []
    for file in files:
        df = pd.read_csv(file)
        df["Experiment"] = os.path.basename(file).replace(".csv", "")
        all_df.append(df)

    df = pd.concat(all_df)

    # --- Loss Plot ---
    df_loss = df.melt(id_vars=["Epoch", "Experiment"],
                      value_vars=["Train Loss", "Validation Loss"],
                      var_name="Metric", value_name="Value")

    fig_loss = px.line(df_loss, x="Epoch", y="Value", color="Experiment", line_dash="Metric",
                       title="Vergleich: Training vs. Validierung (Loss)", log_y=True)
    fig_loss.show()

    # --- Accuracy Plot ---
    if "Accuracy" in df.columns:
        fig_acc = px.line(df, x="Epoch", y="Accuracy", color="Experiment",
                          title="Validierungsgenauigkeit über Epochen")
        fig_acc.show()
    else:
        print("⚠️ Keine Accuracy-Spalte gefunden. Wurde Accuracy im Training geloggt?")

if __name__ == "__main__":

    plot_all_results()