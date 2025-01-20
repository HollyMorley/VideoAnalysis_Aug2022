import pandas as pd
import numpy as np
import os
from Helpers.Config_23 import *

def summarize_run_logs(summary_log_path, error_log_path, output_path="Summary_Metrics.xlsx"):
    """
    1) Computes an overall summary (mean/std/sum of AbsentRuns, UncategorisedRuns, AffectedRuns),
       plus total file count -> "OverallStats" sheet (no chart).
    2) Groups by MouseID -> "ByMouseID" sheet (multi-level columns + chart).
    3) Groups by speed   -> "BySpeed"   sheet (multi-level columns + chart).

    Each grouped sheet includes a bar chart comparing AffectedRuns_Sum among groups.
    """

    # ----------------------------------------------------------------
    # 0) Load the data
    # ----------------------------------------------------------------
    if not os.path.exists(summary_log_path):
        raise FileNotFoundError(f"Could not find summary log: {summary_log_path}")
    if not os.path.exists(error_log_path):
        raise FileNotFoundError(f"Could not find error log: {error_log_path}")

    summary_df = pd.read_csv(summary_log_path)
    error_df   = pd.read_csv(error_log_path)

    # ----------------------------------------------------------------
    # 1) Prepare columns we need
    # ----------------------------------------------------------------
    # Example columns: AbsentRuns = MissingRuns + DroppedRunsPlaceholder
    summary_df["AbsentRuns"] = summary_df["MissingRuns"] + summary_df["DroppedRunsPlaceholder"]

    # UncategorisedRuns = # rows in error_log for each file
    error_count_per_file = error_df.groupby("File").size().reset_index(name="UncategorisedRuns")
    summary_df = pd.merge(summary_df, error_count_per_file, on="File", how="left")
    summary_df["UncategorisedRuns"] = summary_df["UncategorisedRuns"].fillna(0)

    # AffectedRuns = AbsentRuns + UncategorisedRuns (example)
    summary_df["AffectedRuns"] = summary_df["AbsentRuns"] + summary_df["UncategorisedRuns"]

    # Helper to compute mean, std, sum
    def get_stats(series):
        return {"Mean": series.mean(), "Std": series.std(), "Sum": series.sum()}

    # ----------------------------------------------------------------
    # 2) Build an overall summary (no grouping)
    # ----------------------------------------------------------------
    absent_stats = get_stats(summary_df["AbsentRuns"])
    uncat_stats  = get_stats(summary_df["UncategorisedRuns"])
    affect_stats = get_stats(summary_df["AffectedRuns"])

    overall_df = pd.DataFrame({
        "AbsentRuns":       absent_stats,
        "UncategorisedRuns":uncat_stats,
        "AffectedRuns":     affect_stats
    })

    # Add total file count as a row
    total_files = summary_df["File"].nunique()
    overall_df.loc["TotalFiles"] = [total_files, np.nan, np.nan]

    # ----------------------------------------------------------------
    # 3) Build grouped pages: (ByMouseID) and (BySpeed)
    # ----------------------------------------------------------------
    groupings = [
        ("ByMouseID", ["MouseID"]),
        ("BySpeed",   ["speed"]),
    ]

    grouped_dfs = {}  # {sheet_name: DataFrame}

    for sheet_name, group_cols in groupings:
        grouped = summary_df.groupby(group_cols, dropna=False)

        rows = []
        for group_val, subdf in grouped:
            # group_val is the unique MouseID or speed
            # subdf is the subset of summary_df for that group

            # compute stats
            ab_stats = get_stats(subdf["AbsentRuns"])
            un_stats = get_stats(subdf["UncategorisedRuns"])
            af_stats = get_stats(subdf["AffectedRuns"])

            # if group_val is a single value (not a tuple),
            # just convert to str:
            if isinstance(group_val, tuple):
                group_str = "_".join(str(x) for x in group_val)
            else:
                group_str = str(group_val)

            rows.append({
                "Group":               group_str,
                "AbsentRuns_Mean":     ab_stats["Mean"],
                "AbsentRuns_Std":      ab_stats["Std"],
                "AbsentRuns_Sum":      ab_stats["Sum"],
                "Uncategorised_Mean":  un_stats["Mean"],
                "Uncategorised_Std":   un_stats["Std"],
                "Uncategorised_Sum":   un_stats["Sum"],
                "AffectedRuns_Mean":   af_stats["Mean"],
                "AffectedRuns_Std":    af_stats["Std"],
                "AffectedRuns_Sum":    af_stats["Sum"]
            })

        df_group = pd.DataFrame(rows)

        # Make multi-level columns
        # Suppose we want top-level: 'AbsentRuns', 'Uncategorised', 'AffectedRuns'
        # under that: 'Mean','Std','Sum'
        # plus a first column for the 'Group'
        new_cols = [
            (sheet_name,        "Group"),
            ("AbsentRuns",      "Mean"),
            ("AbsentRuns",      "Std"),
            ("AbsentRuns",      "Sum"),
            ("Uncategorised",   "Mean"),
            ("Uncategorised",   "Std"),
            ("Uncategorised",   "Sum"),
            ("AffectedRuns",    "Mean"),
            ("AffectedRuns",    "Std"),
            ("AffectedRuns",    "Sum"),
        ]

        # Map from multi-level -> actual col in df
        mapping = {
            (sheet_name,      "Group"):           "Group",
            ("AbsentRuns",    "Mean"):            "AbsentRuns_Mean",
            ("AbsentRuns",    "Std"):             "AbsentRuns_Std",
            ("AbsentRuns",    "Sum"):             "AbsentRuns_Sum",
            ("Uncategorised", "Mean"):            "Uncategorised_Mean",
            ("Uncategorised", "Std"):             "Uncategorised_Std",
            ("Uncategorised", "Sum"):             "Uncategorised_Sum",
            ("AffectedRuns",  "Mean"):            "AffectedRuns_Mean",
            ("AffectedRuns",  "Std"):             "AffectedRuns_Std",
            ("AffectedRuns",  "Sum"):             "AffectedRuns_Sum",
        }

        # reorder columns
        df_group = df_group[[mapping[col] for col in new_cols]]
        df_group.columns = pd.MultiIndex.from_tuples(new_cols)

        grouped_dfs[sheet_name] = df_group

    # ----------------------------------------------------------------
    # 4) Write everything to Excel. We'll have:
    #    a) OverallStats page (no chart)
    #    b) ByMouseID page    (with chart)
    #    c) BySpeed page      (with chart)
    # ----------------------------------------------------------------
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:

        # (a) OverallStats sheet
        # Single-level columns in overall_df -> no multi-index problem
        overall_df.to_excel(
            writer,
            sheet_name="OverallStats",
            float_format="%.2f",
            index=True  # needed so we see "Mean", "Std", "Sum" as row labels
        )

        # (b) & (c) each grouping
        for sheet_name, df_group in grouped_dfs.items():
            # Because we have multi-level columns, we must do index=True
            # otherwise Pandas triggers NotImplementedError
            df_group.to_excel(
                writer,
                sheet_name=sheet_name,
                index=True,     # we can *try* with or without
                merge_cells=True,
                float_format="%.2f"
            )

            # But if "index=False" raises NotImplementedError for multi-index columns,
            # switch to index=True or flatten columns. Let's do index=True to be safe:
            # (If you get the NotImplementedError, set index=True here)
            # df_group.to_excel(
            #    writer, sheet_name=sheet_name, index=True, merge_cells=True, float_format="%.2f"
            # )

            workbook  = writer.book
            worksheet = writer.sheets[sheet_name]

            # Insert chart comparing "AffectedRuns_Sum"
            # 1) find the column index for ("AffectedRuns", "Sum")
            aff_sum_col_idx = df_group.columns.get_loc(("AffectedRuns", "Sum"))
            # data rows = len(df_group), header row is row=0
            n_rows = len(df_group)

            chart = workbook.add_chart({"type": "column"})
            chart.add_series({
                "name": f"{sheet_name} AffectedRuns_Sum",
                "categories": [sheet_name, 1, 0, n_rows, 0],  # group names in col=0
                "values":     [sheet_name, 1, aff_sum_col_idx, n_rows, aff_sum_col_idx],
            })
            chart.set_title({"name": f"AffectedRuns_Sum by {sheet_name}"})
            worksheet.insert_chart("B15", chart)

    print(f"Created an 'OverallStats' sheet plus '{list(grouped_dfs.keys())}' in '{output_path}'.")
    print("Each grouped sheet has a chart comparing AffectedRuns_Sum among groups.")


if __name__ == "__main__":
    summarize_run_logs(
        summary_log_path=os.path.join(paths['filtereddata_folder'], "run_summary_log.csv"),
        error_log_path=os.path.join(paths['filtereddata_folder'], "error_log.csv"),
        output_path=os.path.join(paths['filtereddata_folder'], "MySummaryResults.xlsx")
    )
