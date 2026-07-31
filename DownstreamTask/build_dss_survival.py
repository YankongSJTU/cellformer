"""
Build DSS (Disease-Specific Survival) and OS (Overall Survival) data
from TCGA GDAC clinical data (Broad Institute).

DSS event definition (proxy): vital_status == "dead" AND person_neoplasm_cancer_status == "with tumor"
OS event definition: vital_status == "dead"

Output format: samplename, time (months), status (0=censored, 1=event)
Compatible with existing CPSformer survival analysis pipeline.

Usage: python build_dss_survival.py [--output_dir DIR] [--cohort COHORT]
"""

import os
import csv
import argparse
from collections import OrderedDict

# ============ Config ============
CLINICAL_DIR = "/data1/TumorGroup/DATA/public_database/TCGA/clinical"
OUTPUT_DIR = "/export/home/kongyan/project/newcellformer/survival_dss"

# PRAD has no standard GDAC clinical data (only 1 patient in PRAD-FFPE).
# Use existing survival data as OS baseline; DSS is same as OS for PRAD.
EXISTING_SURVIVAL_DIR = "/export/home/kongyan/project/cellformer/survival"
PRAD_EXISTING = os.path.join(EXISTING_SURVIVAL_DIR, "PRAD.survival.csv")

# 22 TCGA cohorts used in CPSformer
COHORTS = [
    "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC",
    "OV", "PAAD", "PRAD", "READ", "STAD", "THCA",
]


def parse_gdac_clinical(clin_file):
    """
    Parse GDAC merged_only_clinical_clin_format.txt file.
    Format: row-based (each row is a field, columns are patients).
    Returns dict of {field_name: [values per patient]}
    """
    field_data = OrderedDict()
    with open(clin_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            field_name = parts[0]
            values = parts[1:]
            field_data[field_name] = values
    return field_data


def get_n_patients(field_data):
    """Get number of patients from field data."""
    for k, v in field_data.items():
        return len(v)
    return 0


def build_survival_data(cohort):
    """
    Build OS and DSS survival data for a cohort.
    Returns (os_data, dss_data) as lists of (samplename, time_months, status)
    """
    gdac_dir = os.path.join(
        CLINICAL_DIR,
        f"gdac.broadinstitute.org_{cohort}.Merge_Clinical.Level_1.2016071500.0.0"
    )
    clin_file = os.path.join(gdac_dir, f"{cohort}.merged_only_clinical_clin_format.txt")

    if not os.path.exists(clin_file):
        print(f"  [SKIP] {cohort}: clinical file not found at {clin_file}")
        return [], []

    field_data = parse_gdac_clinical(clin_file)
    n = get_n_patients(field_data)
    print(f"  {cohort}: loaded {n} patients from GDAC clinical data")

    # Extract key fields
    barcode = field_data.get("patient.bcr_patient_barcode", [""] * n)
    vital_status = field_data.get("patient.vital_status", [""] * n)
    days_to_death = field_data.get("patient.days_to_death", ["NA"] * n)
    days_to_last_followup = field_data.get("patient.days_to_last_followup", ["NA"] * n)
    cancer_status = field_data.get("patient.person_neoplasm_cancer_status", [""] * n)

    os_data = []
    dss_data = []

    for i in range(n):
        sid = barcode[i]
        if not sid:
            continue

        # Normalize sample name (TCGA-XX-XXXX format)
        samplename = sid.upper().strip()

        vs = vital_status[i].strip().lower() if vital_status[i] else ""
        dtd = days_to_death[i].strip() if days_to_death[i] else "NA"
        dtlf = days_to_last_followup[i].strip() if days_to_last_followup[i] else "NA"
        cs = cancer_status[i].strip().lower() if cancer_status[i] else ""

        # Parse numeric values
        try:
            dtd_val = float(dtd) if dtd != "NA" and dtd != "" else None
        except ValueError:
            dtd_val = None

        try:
            dtlf_val = float(dtlf) if dtlf != "NA" and dtlf != "" else None
        except ValueError:
            dtlf_val = None

        # Skip if no time info at all
        if dtd_val is None and dtlf_val is None:
            continue

        # ---- OS (Overall Survival) ----
        # Event = dead
        # Time = days_to_death if dead, days_to_last_followup if alive
        os_status = 0
        os_time_days = dtlf_val
        if vs == "dead":
            os_status = 1
            os_time_days = dtd_val  # use days_to_death
        elif vs == "alive":
            os_status = 0
            os_time_days = dtlf_val  # use days_to_last_followup
        else:
            # Unknown vital status - skip
            continue

        if os_time_days is not None and os_time_days > 0:
            os_time_months = round(os_time_days / 30.0, 4)
            os_data.append((samplename, os_time_months, os_status))

        # ---- DSS (Disease-Specific Survival) ----
        # Event = dead AND cancer_status == "with tumor" (proxy for disease-specific death)
        # If dead but tumor free → censor (death from other causes)
        # If alive → censor
        dss_status = 0
        dss_time_days = dtlf_val
        if vs == "dead" and cs == "with tumor":
            # Disease-specific death event
            dss_status = 1
            dss_time_days = dtd_val
        elif vs == "dead":
            # Dead but tumor free or unknown cancer status → censor for DSS
            dss_status = 0
            # Use min(days_to_death, days_to_last_followup) for censoring time
            if dtd_val is not None and dtlf_val is not None:
                dss_time_days = min(dtd_val, dtlf_val)
            elif dtd_val is not None:
                dss_time_days = dtd_val
            else:
                dss_time_days = dtlf_val
        elif vs == "alive":
            dss_status = 0
            dss_time_days = dtlf_val
        else:
            continue

        if dss_time_days is not None and dss_time_days > 0:
            dss_time_months = round(dss_time_days / 30.0, 4)
            dss_data.append((samplename, dss_time_months, dss_status))

    return os_data, dss_data


def write_survival_csv(data, filepath, survival_type="OS"):
    """Write survival data to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["samplename", "time", "status"])
        for samplename, time, status in sorted(data, key=lambda x: x[0]):
            writer.writerow([samplename, time, status])


def copy_existing_survival(existing_file, output_dir, cohort):
    """Copy existing survival data to OS and DSS files."""
    data = []
    with open(existing_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samplename = row['samplename']
            time = float(row['time'])
            status = int(row['status'])
            data.append((samplename, time, status))

    os_path = os.path.join(output_dir, f"{cohort}.os.survival.csv")
    dss_path = os.path.join(output_dir, f"{cohort}.dss.survival.csv")
    write_survival_csv(data, os_path, "OS")
    write_survival_csv(data, dss_path, "DSS")  # For PRAD, DSS=OS (cannot distinguish)

    events = sum(1 for _, _, s in data if s == 1)
    return len(data), events, len(data), events


def main():
    parser = argparse.ArgumentParser(description="Build DSS/OS survival data from TCGA clinical data")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--cohort", type=str, default=None, help="Single cohort to process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cohorts = [args.cohort] if args.cohort else COHORTS

    print("=" * 60)
    print("Building DSS & OS survival data from TCGA GDAC clinical data")
    print("=" * 60)

    total_os = 0
    total_dss = 0
    total_os_events = 0
    total_dss_events = 0

    for cohort in cohorts:
        # Handle PRAD specially - use existing survival data
        if cohort == "PRAD":
            if os.path.exists(PRAD_EXISTING):
                n, events, _, _ = copy_existing_survival(PRAD_EXISTING, args.output_dir, cohort)
                total_os += n
                total_dss += n
                total_os_events += events
                total_dss_events += events
                print(f"  {cohort}: copied existing survival data ({n} samples, {events} events)")
                print(f"    OS: {n} samples ({events} events)")
                print(f"    DSS: {n} samples ({events} events) [same as OS, cause unknown]")
            else:
                print(f"  {cohort}: [SKIP] no GDAC data and existing file not found")
            continue

        os_data, dss_data = build_survival_data(cohort)

        # Write OS file
        os_path = os.path.join(args.output_dir, f"{cohort}.os.survival.csv")
        if os_data:
            write_survival_csv(os_data, os_path, "OS")
            os_events = sum(1 for _, _, s in os_data if s == 1)
            total_os += len(os_data)
            total_os_events += os_events
            print(f"    OS: {len(os_data)} samples ({os_events} events) -> {os_path}")
        else:
            print(f"    OS: no valid data")

        # Write DSS file
        dss_path = os.path.join(args.output_dir, f"{cohort}.dss.survival.csv")
        if dss_data:
            write_survival_csv(dss_data, dss_path, "DSS")
            dss_events = sum(1 for _, _, s in dss_data if s == 1)
            total_dss += len(dss_data)
            total_dss_events += dss_events
            print(f"    DSS: {len(dss_data)} samples ({dss_events} events) -> {dss_path}")
        else:
            print(f"    DSS: no valid data")

    print()
    print("=" * 60)
    print(f"Total: {total_os} OS samples ({total_os_events} events), "
          f"{total_dss} DSS samples ({total_dss_events} events)")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
