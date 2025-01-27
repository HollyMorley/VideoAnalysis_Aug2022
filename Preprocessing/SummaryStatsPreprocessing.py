import pandas as pd
import numpy as np
import os

# We'll use statsmodels for the ANOVA and Tukey HSD
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Ensure xlsxwriter is installed
# You can install it via pip if not already installed:
# pip install xlsxwriter

from Helpers.Config_23 import *

def summarize_run_logs(summary_log_path, error_log_path, output_path="Summary_Metrics.xlsx"):
    """
    Produces an Excel file with sheets:
      1. OverallStats         - Summary for all files, no chart.
      2. ByMouseID            - Grouped by MouseID, bar chart of AffectedRuns mean + std and sem.
      3. BySpeed              - Grouped by Speed (with LowHigh split into Repeats and Extended), bar chart.
      4. ConditionByMouse     - Grouped by speedExpCondition and MouseID, bar chart.
      5. ANOVA                - Two-factor ANOVA results.
      6. TukeyPostHoc         - Tukey's HSD post hoc test results.

    The bar charts show the MEAN of AffectedRuns with error bars from both STD and SEM.
    """

    # ----------------------------------------------------------------
    # 0) Load CSV data
    # ----------------------------------------------------------------
    if not os.path.exists(summary_log_path):
        raise FileNotFoundError(f"Could not find summary log: {summary_log_path}")
    if not os.path.exists(error_log_path):
        raise FileNotFoundError(f"Could not find error log: {error_log_path}")

    summary_df = pd.read_csv(summary_log_path)
    error_df = pd.read_csv(error_log_path)

    print("Loaded summary_df shape =", summary_df.shape)
    print("summary_df columns =", summary_df.columns.tolist())
    print("Loaded error_df shape   =", error_df.shape)
    print("error_df columns   =", error_df.columns.tolist())

    # Prepare additional columns
    summary_df["AbsentRuns"] = summary_df["MissingRuns"] + summary_df["DroppedRunsPlaceholder"]
    error_count_per_file = error_df.groupby("File").size().reset_index(name="UncategorisedRuns")
    summary_df = pd.merge(summary_df, error_count_per_file, on="File", how="left")
    summary_df["UncategorisedRuns"] = summary_df["UncategorisedRuns"].fillna(0)
    summary_df["AffectedRuns"] = summary_df["AbsentRuns"] + summary_df["UncategorisedRuns"]

    # Helper function to calculate statistics
    def calculate_stats(series):
        n = len(series)
        mean = series.mean()
        std = series.std()
        sem = std / np.sqrt(n) if n > 1 else 0  # Added SEM calculation
        return {"mean": mean, "std": std, "sem": sem, "sum": series.sum()}

    # ----------------------------------------------------------------
    # 1) OverallStats sheet (no grouping)
    # ----------------------------------------------------------------
    overall_stats = {
        "AbsentRuns": calculate_stats(summary_df["AbsentRuns"]),
        "UncategorisedRuns": calculate_stats(summary_df["UncategorisedRuns"]),
        "AffectedRuns": calculate_stats(summary_df["AffectedRuns"]),
    }
    overall_df = pd.DataFrame(overall_stats).T

    # Add a row for TotalFiles
    overall_df.loc["TotalFiles"] = {
        "mean": summary_df["File"].nunique(),  # Total unique files
        "std": np.nan,
        "sem": np.nan,
        "sum": np.nan,
    }

    # ----------------------------------------------------------------
    # 2) ByMouseID sheet
    # ----------------------------------------------------------------
    grouped_by_mouse = summary_df.groupby("MouseID")
    bymouse_data = []
    for mouse_id, group in grouped_by_mouse:
        stats = {
            "AbsentRuns": calculate_stats(group["AbsentRuns"]),
            "UncategorisedRuns": calculate_stats(group["UncategorisedRuns"]),
            "AffectedRuns": calculate_stats(group["AffectedRuns"]),
        }
        bymouse_data.append({
            "Group": mouse_id,
            "AbsentRuns_Mean": stats["AbsentRuns"]["mean"],
            "AbsentRuns_Std": stats["AbsentRuns"]["std"],
            "AbsentRuns_SEM": stats["AbsentRuns"]["sem"],  # Added SEM
            "AbsentRuns_Sum": stats["AbsentRuns"]["sum"],
            "Uncategorised_Mean": stats["UncategorisedRuns"]["mean"],
            "Uncategorised_Std": stats["UncategorisedRuns"]["std"],
            "Uncategorised_SEM": stats["UncategorisedRuns"]["sem"],  # Added SEM
            "Uncategorised_Sum": stats["UncategorisedRuns"]["sum"],
            "AffectedRuns_Mean": stats["AffectedRuns"]["mean"],
            "AffectedRuns_Std": stats["AffectedRuns"]["std"],
            "AffectedRuns_SEM": stats["AffectedRuns"]["sem"],  # Added SEM
            "AffectedRuns_Sum": stats["AffectedRuns"]["sum"],
        })
    bymouse_df = pd.DataFrame(bymouse_data)

    # ----------------------------------------------------------------
    # 3) BySpeed sheet
    # ----------------------------------------------------------------
    summary_df["SpeedGroup"] = summary_df.apply(
        lambda row: f"LowHigh-{row['repeat_extend']}" if row["speed"] == "LowHigh" else row["speed"], axis=1
    )

    grouped_by_speed = summary_df.groupby("SpeedGroup")
    byspeed_data = []
    for speed, group in grouped_by_speed:
        stats = {
            "AbsentRuns": calculate_stats(group["AbsentRuns"]),
            "UncategorisedRuns": calculate_stats(group["UncategorisedRuns"]),
            "AffectedRuns": calculate_stats(group["AffectedRuns"]),
        }
        byspeed_data.append({
            "Group": speed,
            "AbsentRuns_Mean": stats["AbsentRuns"]["mean"],
            "AbsentRuns_Std": stats["AbsentRuns"]["std"],
            "AbsentRuns_SEM": stats["AbsentRuns"]["sem"],  # Added SEM
            "AbsentRuns_Sum": stats["AbsentRuns"]["sum"],
            "Uncategorised_Mean": stats["UncategorisedRuns"]["mean"],
            "Uncategorised_Std": stats["UncategorisedRuns"]["std"],
            "Uncategorised_SEM": stats["UncategorisedRuns"]["sem"],  # Added SEM
            "Uncategorised_Sum": stats["UncategorisedRuns"]["sum"],
            "AffectedRuns_Mean": stats["AffectedRuns"]["mean"],
            "AffectedRuns_Std": stats["AffectedRuns"]["std"],
            "AffectedRuns_SEM": stats["AffectedRuns"]["sem"],  # Added SEM
            "AffectedRuns_Sum": stats["AffectedRuns"]["sum"],
        })
    byspeed_df = pd.DataFrame(byspeed_data)

    # ----------------------------------------------------------------
    # 4) Add MultiIndex Columns
    # ----------------------------------------------------------------
    def build_multiindex_df(df, name):
        cols = [
            (name, "Group"),
            ("AbsentRuns", "Mean"), ("AbsentRuns", "Std"), ("AbsentRuns", "SEM"), ("AbsentRuns", "Sum"),
            ("Uncategorised", "Mean"), ("Uncategorised", "Std"), ("Uncategorised", "SEM"), ("Uncategorised", "Sum"),
            ("AffectedRuns", "Mean"), ("AffectedRuns", "Std"), ("AffectedRuns", "SEM"), ("AffectedRuns", "Sum"),
        ]
        df.columns = pd.MultiIndex.from_tuples(cols)
        return df

    bymouse_multi = build_multiindex_df(bymouse_df, "ByMouseID")
    byspeed_multi = build_multiindex_df(byspeed_df, "BySpeed")

    # ----------------------------------------------------------------
    # 5) ANOVA + Post Hoc
    # ----------------------------------------------------------------
    summary_df["MouseID"] = summary_df["MouseID"].astype(str)
    summary_df["Group"] = summary_df["SpeedGroup"] + "__" + summary_df["MouseID"]

    anova_df = summary_df.dropna(subset=["AffectedRuns", "SpeedGroup", "MouseID"])
    model = ols("AffectedRuns ~ C(SpeedGroup) * C(MouseID)", data=anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    tukey = pairwise_tukeyhsd(
        endog=anova_df["AffectedRuns"],
        groups=anova_df["Group"],
        alpha=0.05
    )
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

    # ----------------------------------------------------------------
    # 6) Write to Excel with charts
    # ----------------------------------------------------------------
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Write existing sheets
        overall_df.to_excel(writer, sheet_name="OverallStats", float_format="%.2f")
        bymouse_multi.to_excel(writer, sheet_name="ByMouseID", merge_cells=True)
        byspeed_multi.to_excel(writer, sheet_name="BySpeed", merge_cells=True)
        anova_table.to_excel(writer, sheet_name="ANOVA")
        tukey_df.to_excel(writer, sheet_name="TukeyPostHoc")

        workbook = writer.book

        # Restore bar charts for ByMouseID and BySpeed
        for sheet_name, df in [("ByMouseID", bymouse_multi), ("BySpeed", byspeed_multi)]:
            worksheet = writer.sheets[sheet_name]
            for error_type in ["Std", "SEM"]:
                chart = workbook.add_chart({"type": "column"})
                n_rows = len(df)

                mean_col = df.columns.get_loc(("AffectedRuns", "Mean"))
                error_col = df.columns.get_loc(("AffectedRuns", error_type))

                chart.add_series({
                    "name": f"Mean ± {error_type}",
                    "categories": [sheet_name, 1, 0, n_rows, 0],
                    "values": [sheet_name, 1, mean_col, n_rows, mean_col],
                    "y_error_bars": {
                        "type": "custom",
                        "plus_values": [sheet_name, 1, error_col, n_rows, error_col],
                        "minus_values": [sheet_name, 1, error_col, n_rows, error_col],
                    },
                })

                chart.set_title({"name": f"AffectedRuns Mean ± {error_type}"})
                chart.set_x_axis({"name": "Group"})
                chart.set_y_axis({"name": "Mean AffectedRuns"})
                worksheet.insert_chart("J2" if error_type == "Std" else "J20", chart)

        # Add a new sheet with plots for ByCondition
        worksheet = writer.book.add_worksheet("ByCondition")
        writer.sheets["ByCondition"] = worksheet

        grouped_conditions = summary_df.groupby("SpeedGroup")
        start_row = 0  # Start row for writing data and inserting charts

        for condition, condition_data in grouped_conditions:
            stats = condition_data.groupby("MouseID")["AffectedRuns"].agg(
                mean="mean", sem=lambda x: x.std() / np.sqrt(len(x))
            ).reset_index()

            # Write data to the worksheet
            worksheet.write(start_row, 0, f"Condition: {condition}")
            worksheet.write(start_row + 1, 0, "MouseID")
            worksheet.write(start_row + 1, 1, "Mean AffectedRuns")
            worksheet.write(start_row + 1, 2, "SEM")

            for i, row_data in stats.iterrows():
                worksheet.write(start_row + 2 + i, 0, row_data["MouseID"])
                worksheet.write(start_row + 2 + i, 1, clean_value(row_data["mean"]))
                worksheet.write(start_row + 2 + i, 2, clean_value(row_data["sem"]))

            # Add a bar chart for the condition
            chart = writer.book.add_chart({"type": "column"})
            chart.add_series({
                "name": f"Condition: {condition}",
                "categories": [f"ByCondition", start_row + 2, 0, start_row + 2 + len(stats) - 1, 0],
                "values": [f"ByCondition", start_row + 2, 1, start_row + 2 + len(stats) - 1, 1],
                "y_error_bars": {
                    "type": "custom",
                    "plus_values": [f"ByCondition", start_row + 2, 2, start_row + 2 + len(stats) - 1, 2],
                    "minus_values": [f"ByCondition", start_row + 2, 2, start_row + 2 + len(stats) - 1, 2],
                },
            })
            chart.set_title({"name": f"AffectedRuns: {condition}"})
            chart.set_x_axis({"name": "MouseID"})
            chart.set_y_axis({"name": "Mean AffectedRuns"})
            worksheet.insert_chart(start_row, 4, chart)

            # Update start_row for the next condition
            start_row += len(stats) + 6

    print(f"Excel file updated with ByCondition plots and restored charts at: {output_path}")

    print(f"Excel file created at: {output_path}")

def clean_value(value):
    if pd.isna(value) or np.isinf(value):  # Check for NaN or inf
        return 0  # Replace with 0
    return value


if __name__ == "__main__":
    summarize_run_logs(
        summary_log_path=os.path.join(paths["filtereddata_folder"], "run_summary_log.csv"),
        error_log_path=os.path.join(paths["filtereddata_folder"], "error_log.csv"),
        output_path=os.path.join(paths["filtereddata_folder"], "MySummaryResults.xlsx")
    )
