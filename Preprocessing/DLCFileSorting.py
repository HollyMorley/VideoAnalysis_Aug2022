import os
import shutil

dlc_dir = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round1"  # Adjust if needed
dlc_dest = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round3_Aug24"

exp_cats = {
    'APAChar_LowHigh': {
        'Repeats': {
            'Wash': {
                'Exp': {
                    'Day1': {'A': ['20230306'], 'B': ['20230309']},
                    'Day2': {'A': ['20230308'], 'B': ['20230312']},
                    'Day3': {'A': ['20230310'], 'B': ['20230314']},
                },
                'Washout': {
                    'Day1': {'A': ['20230307'], 'B': ['20230310']},
                    'Day2': {'A': ['20230309'], 'B': ['20230313']},
                    'Day3': {'A': ['20230312'], 'B': ['20230315']},
                }
            }
        },
        'Extended': {
            'Day1': {'A': ['20230316'], 'B': ['20230403']},
            'Day2': {'A': ['20230317'], 'B': ['20230404']},
            'Day3': {'A': ['20230318'], 'B': ['20230405']},
            'Day4': {'A': ['20230319'], 'B': ['20230406']}
        }
    },
    'APAChar_HighLow': {
        'Extended': {
            'Day1': {'A': ['20230325'], 'B': ['20230412']},
            'Day2': {'A': ['20230326'], 'B': ['20230413']},
            'Day3': {'A': ['20230327'], 'B': ['20230414']},
            'Day4': {'A': ['20230328'], 'B': ['20230415']}
        }
    },
    'APAChar_LowMid': {
        'Extended': {
            'Day1': {'A': ['20230407'], 'B': ['20230320']},
            'Day2': {'A': ['20230408'], 'B': ['20230321']},
            'Day3': {'A': ['20230409'], 'B': ['20230322']},
            'Day4': {'A': ['20230410'], 'B': ['20230323']}
        }
    },
    'PerceptionTest': {'A': ['20230411'], 'B': ['20230416']},
    'VMT_LowHigh': {
        'pd': {
            'Basic': {
                'Prep': {'A': ['20230420'], 'B': ['20230428']},
                'Experiment': {'A': ['20230421'], 'B': ['20230427']}
            },
            'Extreme': {
                'Prep': {'A': ['20230513'], 'B': ['20230514']},
                'Experiment': {'A': ['20230514'], 'B': ['20230515']}
            }
        },
        'ac': {
            'Basic': {
                'Prep': {'A': [''], 'B': ['20230503']},
                'Experiment': {'A': [''], 'B': ['20230504']}
            },
            'Extreme': {
                'Prep': {'A': ['20230515'], 'B': ['20230516']},
                'Experiment': {'A': ['20230516'], 'B': ['20230517']}
            }
        }
    },
    'VMT_HighLow': {
        'pd': {
            'Basic': {
                'Prep': {'A': [''], 'B': ['']},
                'Experiment': {'A': [''], 'B': ['']}
            }
        },
        'ac': {
            'Basic': {
                'Prep': {'A': [''], 'B': ['20230505']},
                'Experiment': {'A': [''], 'B': ['20230506']}
            }
        }
    }
}

MouseIDs = {'A': ['1035243', '1035244', '1035245', '1035246', '1035249', '1035250'],
            'B': ['1035297', '1035298', '1035299', '1035301', '1035302']
            }

# Function to create directories if they don't exist
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory: {path}')

# Function to find and copy files based on exp_cats and MouseIDs
def copy_files(src_dir, dest_dir, exp_cats, MouseIDs):
    for category, subcategories in exp_cats.items():
        if isinstance(subcategories, dict):
            for subcategory, phases in subcategories.items():
                if isinstance(phases, dict):
                    for phase, days in phases.items():
                        if isinstance(days, dict):
                            for day, mice_dates in days.items():
                                if isinstance(mice_dates, dict):
                                    for mouse_group, dates in mice_dates.items():
                                        if mouse_group in MouseIDs:
                                            for date in dates:
                                                print(f'Processing: {category}, {subcategory}, {phase}, {day}, {mouse_group}, {date}')
                                                copy_files_for_mouse_group(src_dir, dest_dir, category, subcategory, phase, day, mouse_group, date, MouseIDs)
                                else:
                                    for mouse_group in ['A', 'B']:
                                        if mouse_group in MouseIDs:
                                            for date in mice_dates:
                                                print(f'Processing: {category}, {subcategory}, {mouse_group}, {date}')
                                                copy_files_for_mouse_group(src_dir, dest_dir, category, subcategory, '', '', mouse_group, date, MouseIDs)
                else:
                    for mouse_group in ['A', 'B']:
                        if mouse_group in MouseIDs:
                            for date in phases:
                                print(f'Processing: {category}, {subcategory}, {mouse_group}, {date}')
                                copy_files_for_mouse_group(src_dir, dest_dir, category, subcategory, '', '', mouse_group, date, MouseIDs)
        else:
            for mouse_group in ['A', 'B']:
                if mouse_group in MouseIDs:
                    for date in subcategories:
                        print(f'Processing: {category}, {mouse_group}, {date}')
                        copy_files_for_mouse_group(src_dir, dest_dir, category, '', '', '', mouse_group, date, MouseIDs)

# Helper function to copy files for a specific mouse group and date
def copy_files_for_mouse_group(src_dir, dest_dir, category, subcategory, phase, day, mouse_group, date, MouseIDs):
    final_dest_dir = os.path.join(dest_dir, category, subcategory, phase, day, mouse_group)
    ensure_dir_exists(final_dest_dir)

    file_found = False
    for mouse_id in MouseIDs[mouse_group]:
        search_pattern = f"{mouse_id}_{date}.h5"
        for file_name in os.listdir(src_dir):
            if file_name.endswith('.h5') and search_pattern in file_name:
                src_file = os.path.join(src_dir, file_name)
                dest_file = os.path.join(final_dest_dir, file_name)
                shutil.copy(src_file, dest_file)
                print(f'Copied: {src_file} to {dest_file}')
                file_found = True

    if not file_found:
        print(f"No files found for {mouse_group} on {date}")

# Run the file copying process
copy_files(dlc_dir, dlc_dest, exp_cats, MouseIDs)
