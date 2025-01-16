import pandas as pd
import numpy as np
import os
from Helpers.Config_23 import *

def summarize_run_logs(
    summary_log_path,
    error_log_path,
    output_path="Summary_Metrics.xlsx"  # <--- changed default to .xlsx
):
    """
    Summarize metrics from run_summary_log.csv and error_log.csv.

    Now exports multi-level headers and can embed plots into the .xlsx output.
    """

    if not os.path.exists(summary_log_path):
        raise FileNotFoundError(f"Could not find summary log: {summary_log_path}")
    if not os.path.exists(error_log_path):
        raise FileNotFoundError(f"Could not find error log: {error_log_path}")

    summary_df = pd.read_csv(summary_log_path)
    error_df   = pd.read_csv(error_log_path)

    # --- Create new columns ---
    summary_df["AbsentRuns"] = summary_df["MissingRuns"] + summary_df["DroppedRunsPlaceholder"]

    error_count_per_file = error_df.groupby("File").size().reset_index(name="UncategorisedRuns")
    summary_df = pd.merge(summary_df, error_count_per_file, on="File", how="left")
    summary_df["UncategorisedRuns"] = summary_df["UncategorisedRuns"].fillna(0)

    summary_df["AffectedRuns"] = summary_df["AbsentRuns"] + summary_df["UncategorisedRuns"]

    # --- Basic stats function ---
    def get_stats(series):
        return {
            "Mean": series.mean(),
            "Std":  series.std(),
            "Sum":  series.sum()
        }

    # Overall stats
    absent_stats = get_stats(summary_df["AbsentRuns"])
    uncat_stats  = get_stats(summary_df["UncategorisedRuns"])
    affec_stats  = get_stats(summary_df["AffectedRuns"])

    overall_df = pd.DataFrame({
        "AbsentRuns":       absent_stats,
        "UncategorisedRuns":uncat_stats,
        "AffectedRuns":     affec_stats
    })

    total_files = summary_df["File"].nunique()
    overall_df.loc["TotalFiles"] = [total_files, np.nan, np.nan]

    # Grouped stats
    group_cols = ["MouseID", "exp", "speed"]
    grouped = summary_df.groupby(group_cols, dropna=False)

    group_stats_list = []
    for group_name, group_data in grouped:
        sub_absent_stats = get_stats(group_data["AbsentRuns"])
        sub_uncat_stats  = get_stats(group_data["UncategorisedRuns"])
        sub_affect_stats = get_stats(group_data["AffectedRuns"])

        gf_count = group_data["File"].nunique()
        gf_percent = (gf_count / total_files) * 100

        row_dict = {
            "Group": group_name,
            "NFiles": gf_count,
            "NFiles_%": gf_percent,
            "AbsentRuns_Mean":  sub_absent_stats["Mean"],
            "AbsentRuns_Std":   sub_absent_stats["Std"],
            "AbsentRuns_Sum":   sub_absent_stats["Sum"],
            "UncatRuns_Mean":   sub_uncat_stats["Mean"],
            "UncatRuns_Std":    sub_uncat_stats["Std"],
            "UncatRuns_Sum":    sub_uncat_stats["Sum"],
            "AffectedRuns_Mean":sub_affect_stats["Mean"],
            "AffectedRuns_Std": sub_affect_stats["Std"],
            "AffectedRuns_Sum": sub_affect_stats["Sum"]
        }
        group_stats_list.append(row_dict)

    group_stats_df = pd.DataFrame(group_stats_list)

    # ----------------------------------------------------------------
    #  Make multi-level columns for group_stats_df so it looks nicer.
    # ----------------------------------------------------------------
    # Suppose we want something like:
    #   top-level = 'AbsentRuns', sub-level = 'Mean', 'Std', 'Sum'
    #   top-level = 'UncatRuns',  sub-level = 'Mean', 'Std', 'Sum'
    #   top-level = 'AffectedRuns', sub-level = 'Mean', 'Std', 'Sum'
    #
    # We'll define a list of (level1, level2) column tuples:
    new_cols = [
        ("Group",            ""),  # no second level
        ("NFiles",           ""),
        ("NFiles",           "%"),
        ("AbsentRuns",       "Mean"),
        ("AbsentRuns",       "Std"),
        ("AbsentRuns",       "Sum"),
        ("UncategorisedRuns","Mean"),
        ("UncategorisedRuns","Std"),
        ("UncategorisedRuns","Sum"),
        ("AffectedRuns",     "Mean"),
        ("AffectedRuns",     "Std"),
        ("AffectedRuns",     "Sum"),
    ]

    # Now, we match each tuple to the original column name:
    mapping = {
        ("Group", ""):                "Group",
        ("NFiles", ""):               "NFiles",
        ("NFiles", "%"):              "NFiles_%",
        ("AbsentRuns", "Mean"):       "AbsentRuns_Mean",
        ("AbsentRuns", "Std"):        "AbsentRuns_Std",
        ("AbsentRuns", "Sum"):        "AbsentRuns_Sum",
        ("UncategorisedRuns","Mean"): "UncatRuns_Mean",
        ("UncategorisedRuns","Std"):  "UncatRuns_Std",
        ("UncategorisedRuns","Sum"):  "UncatRuns_Sum",
        ("AffectedRuns","Mean"):      "AffectedRuns_Mean",
        ("AffectedRuns","Std"):       "AffectedRuns_Std",
        ("AffectedRuns","Sum"):       "AffectedRuns_Sum",
    }

    # Reorder the columns in group_stats_df to match 'new_cols'
    group_stats_df = group_stats_df[[mapping[col] for col in new_cols]]

    # Create a MultiIndex from 'new_cols'
    group_stats_df.columns = pd.MultiIndex.from_tuples(new_cols)

    # ----------------------------------------------------------------
    #  Finally: write to an Excel file, with multi-level headers
    # ----------------------------------------------------------------
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # 1) Write the "overall" stats:
        overall_df.to_excel(
            writer,
            sheet_name="OverallStats",
            float_format="%.2f"
        )

        # 2) Write the group stats with multi-level columns:
        group_stats_df.to_excel(
            writer,
            sheet_name="GroupedStats",
            float_format="%.2f",
            merge_cells=True  # merges the top level as you'd expect
        )

        # Optionally, you can access the workbook & worksheet objects:
        workbook  = writer.book
        worksheet = writer.sheets["GroupedStats"]

        # We can do some cell formatting if desired (e.g. widths):
        worksheet.set_column("A:A", 30)  # e.g. widen the first column

        # -------------------------------------------------------------
        # (Optional) 2) Insert a plot into the Excel file
        # -------------------------------------------------------------
        chart = workbook.add_chart({"type": "column"})
        # For example, we’ll plot “AffectedRuns_Sum” vs group rows

        # We know group_stats_df starts in row=1, col=0.
        # Let's find which column is AffectedRuns_Sum:
        # Because we used a multi-level column, we do something like:
        #   top-level = "AffectedRuns", sub-level="Sum"
        # to find that column index. Let's do it dynamically:
        col_idx = group_stats_df.columns.get_loc(("AffectedRuns","Sum"))
        n_rows  = len(group_stats_df) + 1  # +1 for the header row
        # Add series referencing the data in the sheet
        # "GroupedStats" => the name of the Excel sheet
        # data starts at row=1, column=col_idx, ends at row=n_rows-1 (plus the header offset)
        # your X categories are in col=0 (the “Group” column)
        # Remember: "A1" is row=0, col=0 in xlsxwriter coords
        # We'll build range strings carefully:
        series_dict = {
            "name":       "AffectedRuns Sum",
            "categories": ["GroupedStats", 1, 0, n_rows, 0],       # column 0 (Group)
            "values":     ["GroupedStats", 1, col_idx, n_rows, col_idx],
        }
        chart.add_series(series_dict)
        chart.set_title({"name": "AffectedRuns_Sum by Group"})

        # Insert chart into the sheet (B15, as an example)
        worksheet.insert_chart("B15", chart)

        # Done.
        print(f"Multi-level group stats, plus a plot, have been saved to '{output_path}'.")


if __name__ == "__main__":
    summarize_run_logs(
        summary_log_path=os.path.join(paths['filtereddata_folder'], "run_summary_log.csv"),
        error_log_path=os.path.join(paths['filtereddata_folder'], "error_log.csv"),
        # Make sure to end with .xlsx if you want multi-level headers + chart
        output_path=os.path.join(paths['filtereddata_folder'], "MySummaryResults.xlsx")
    )
