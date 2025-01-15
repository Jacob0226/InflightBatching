import argparse
import json
import os
import sys
import pandas as pd
import xlsxwriter
import os

Dur = "3m"
I_Len = ["i2500", "i5500", "i11000"]
N_User = [1, 8, 16, 24, 32, 40, 48, 56, 64, 128, 256]
# Get test names. e.g, 3m_i2500_01user
metrics = ["#Req", "E2E", "TTFT", "TPOT"]
test_cases = []
for ilen in I_Len:
    for n in N_User:
        combined = "_".join([Dur, ilen, "{:02d}".format(n)+"user"])
        test_cases.append(combined)

def FillInExcelTemplate(row, col, model, N_User, worksheet):
    worksheet.write(row, col, model)
    worksheet.write(row+2, col+1, "i2500-o350")
    worksheet.write(row+2, col+5, "i5500-o350")
    worksheet.write(row+2, col+9, "i11000-o350")

    for i in range(3):
        offset = i*4
        worksheet.write(row+3, col+1+offset, "#Req")
        worksheet.write(row+3, col+2+offset, "E2E(s)")
        worksheet.write(row+3, col+3+offset, "TTFT(s)")
        worksheet.write(row+3, col+4+offset, "TPOT(s)")
    
    # Column title
    worksheet.write(row+2, col, "ilen/olen")
    worksheet.write(row+3, col, "#User")
    row_offset = 4
    for n_user in N_User:
        worksheet.write(row+row_offset, col, str(n_user))
        row_offset+=1

def WriteExcel(worksheet, data, row, col, table_name):
    # Excel (x,y) position
    row_offset_nuser = {n_user: offset + 4 for offset, n_user in enumerate(N_User)}
    col_offset_ilen={"i2500":1, "i5500":5, "i11000":9}

    FillInExcelTemplate(row, col, table_name, N_User, worksheet)

    for ilen in I_Len:
        for n_user in N_User:
            n_user_str = "{:02}".format(n_user) + "user"
            dur_ilen_user = '_'.join([Dur, ilen, n_user_str])
            if dur_ilen_user in data:
                r_offset = row_offset_nuser[n_user]
                c_offset = col_offset_ilen[ilen]
                worksheet.write(row+r_offset, col+c_offset+0, data[dur_ilen_user]["#Req"])
                worksheet.write(row+r_offset, col+c_offset+1, data[dur_ilen_user]["E2E"])
                worksheet.write(row+r_offset, col+c_offset+2, data[dur_ilen_user]["TTFT"])
                worksheet.write(row+r_offset, col+c_offset+3, data[dur_ilen_user]["TPOT"])       
    row+=17   
    return row, col           

def ProcessData(data_fodler, excel_folder, excel_filename):
    workbook = xlsxwriter.Workbook(excel_filename)
    worksheet = workbook.add_worksheet()

    benchmark_folder = os.path.join(data_fodler, "benchmark")
    files = os.listdir(benchmark_folder)
    jsons = [os.path.join(benchmark_folder, file) for file in files if file.endswith('.json')]

    # Load json files
    json_data = []
    for j_file in jsons:
        with open(j_file, 'r') as f:
            data = json.load(f)  # Load JSON data as a Python dictionary
            if len(data['vLLM']) > 0:
                data_server_type = data['vLLM']
                print(f'load vLLM json data {j_file}')
            elif len(data['Triton']) > 0:
                data_server_type = data['Triton']
                print(f'load Triton json data {j_file}')
            else:
                print(f"The json file {j_file} doesn't have data in vLLM or Triton. Skip  {j_file} ...")
                continue
            
            keys = list(data_server_type.keys())
            if len(keys) != 1:
                print(f"Error. The number of keys in vLLM or Trtiton should be 1. skip {j_file} ...")
            json_data.append(data_server_type[keys[0]])


    # Init sum_up data
    sum_data = {}
    for test_case in test_cases:
        sum_data[test_case] = {}
        for metric in metrics:
            sum_data[test_case][metric] = 0
        sum_data[test_case]["#Data"] = 0 # Save for future usage

    # sum up metrics
    for data in json_data:  
        for test_case in test_cases:
            if test_case in data:
                for metric in metrics:
                    sum_data[test_case][metric] += data[test_case][metric]

    # Average the E2E, TTFT, TPOT. Leave #Req not changed.
    for test_case in test_cases:
        for metric in {'E2E', 'TTFT', 'TPOT'}:
            # The first '/ len(json_data)' gives average latency of 1 server. Unit: ms/server/req
            # The second '/ len(json_data)' gives average latency of 1 request. Unit: ms/req
            sum_data[test_case][metric] = sum_data[test_case][metric] / len(json_data) / len(json_data)

    # Write excel
    table_name = data_fodler.split("/")[1] # MultiServer_vLLM/8B_BF16_8xTP1 -> 8B_BF16_8xTP1
    row, col = 0, 0 
    row, col = WriteExcel(worksheet, sum_data, row, col, table_name+"_ALL")
    for idx, data in enumerate(json_data):
        row, col = WriteExcel(worksheet, data, row, col, table_name + f"_server{idx+1}")

    print(f"Save to {excel_filename}\n")
    workbook.close()
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the benchmark result from Serving tests.")
    parser.add_argument(
        "-r",
        type=str,
        help="The root folder",
        required=True,
    )
    args = parser.parse_args()

    items = os.listdir(args.r)
    excel_folder = os.path.join(args.r, "excel")
    os.makedirs(excel_folder, exist_ok=True)
    folders = [item for item in items if os.path.isdir(os.path.join(args.r, item)) and item != "excel"]
    for folder in folders:
        print(f"Process {folder}")
        excel_filename = os.path.join(excel_folder, folder+'.xlsx')
        ProcessData(os.path.join(args.r, folder), excel_folder, excel_filename)

'''
python GenerateExcel_MultiServer.py -r MultiServer_vLLM


'''