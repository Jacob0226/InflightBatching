import argparse
import json
import os
import sys
import pandas as pd
import csv  # Import the csv module
import os

ILEN=["i2000", "i4000", "i8500"] # Note: edit the sizes

def FillInCsvHeader(concurrency_list, folder, csv_writer):
    csv_writer.writerow([folder])
    header = ["#Concur"]
    for ilen in ILEN:
        header.extend([f"{ilen}_TTFT(ms)", f"{ilen}_TPOT(s)", f"{ilen}_E2E(ms)"])
    csv_writer.writerow(header)

def ProcessData(args):
    output_filename = args.o
    concurrency_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    folder = args.folder

    with open(output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        FillInCsvHeader(concurrency_list, folder, csv_writer)

        for concur in concurrency_list:
            row_data = [concur]
            num_prompts = concur*20
            for ilen in ILEN:
                benchamrk_log = os.path.join(folder, f"{ilen}_o{200}_c{concur}_p{num_prompts}.log")
                with open(benchamrk_log, 'r') as file:
                    ttft, tpot, e2el= 'None', 'None', 'None'
                    for line in file:
                        if "Mean TTFT (ms):" in line:
                            ttft = float(line.split("Mean TTFT (ms):")[-1].strip())
                        elif "Mean TPOT (ms):" in line:
                            tpot = float(line.split("Mean TPOT (ms):")[-1].strip())
                        elif "Mean E2EL (ms):" in line:
                            e2el= float(line.split("Mean E2EL (ms):")[-1].strip())
                row_data.extend([ttft, tpot, e2el])       
            csv_writer.writerow(row_data)
    print(f"Successfully save into {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the benchmark result from Serving tests.")
    parser.add_argument(
        "--folder",
        type=str,
        help="The benchmark folder",
        required=True,
    )
    parser.add_argument(
        "--o",
        type=str,
        help="out csv filename",
        default="result.csv"
    )

    args = parser.parse_args()
    ProcessData(args)

'''
python Generate_Benchmark_Serving_Excel.py \
    --folder Result/0516/DiffSeed/Llama-3.1-8B/ \
    --o Result/0516/DiffSeed/Llama-3.1-8B.csv

'''