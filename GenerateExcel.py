import argparse
import json
import os
import sys
import pandas as pd
import xlsxwriter
import os

def FillInExcelTemplate(row, col, model, N_User, worksheet):
    worksheet.write(row, col, model)
    worksheet.write(row+1, col+1, "i2500-o350")
    worksheet.write(row+1, col+5, "i5500-o350")
    worksheet.write(row+1, col+9, "i11000-o350")

    for i in range(3):
        offset = i*4
        worksheet.write(row+2, col+1+offset, "#Req")
        worksheet.write(row+2, col+2+offset, "E2E(s)")
        worksheet.write(row+2, col+3+offset, "TTFT(s)")
        worksheet.write(row+2, col+4+offset, "TPOT(s)")
    
    # Column title
    worksheet.write(row+1, col, "ilen/olen")
    worksheet.write(row+2, col, "#User")
    row_offset = 3
    for n_user in N_User:
        worksheet.write(row+row_offset, col, str(n_user))
        row_offset+=1


def ProcessData(args):
    workbook = xlsxwriter.Workbook(args.o)
    worksheet = workbook.add_worksheet()

    Models = [
        "Llama-3.1-8B_float16_TP1",
        "Llama-3.1-8B-Instruct-FP8-KV_float16_TP1",
        "Llama-3.1-8B_float16_TP2",
        "Llama-3.1-8B-Instruct-FP8-KV_float16_TP2",
        "Llama-3.1-70B_float16_TP4",
        "Llama-3.1-70B-Instruct-FP8-KV_float16_TP4"
    ]

    Dur = "3m"
    I_Len = ["i2500", "i5500", "i11000"]
    N_User = [1, 8, 16, 24, 32, 40, 48, 56, 64]

    # Excel (x,y) position
    row, col = 0, 0
    row_offset_nuser = {n_user: offset + 3 for offset, n_user in enumerate(N_User)}
    col_offset_ilen={"i2500":1, "i5500":5, "i11000":9}

    with open(args.i, 'r') as f:
        data = json.load(f)  # Load JSON data as a Python dictionary
    if data['vLLM'] == {}: # Empty dictionary
        data = data['Triton']
    elif data['Triton'] == {}: # Empty dictionary
        data = data["vLLM"]
    
    for model in Models:
        if model not in data:
            print(f"model {model} not in json. Skip...")
            continue
        # Fill in the excel table 
        FillInExcelTemplate(row, col, model, N_User, worksheet)

        for ilen in I_Len:
            for n_user in N_User:
                n_user_str = "{:02}".format(n_user) + "user"
                dur_ilen_user = '/'.join([Dur, ilen, n_user_str])
                if dur_ilen_user not in data[model]:
                    print(f"{dur_ilen_user} not in {model}. Skip...")
                    r_offset = row_offset_nuser[n_user]
                    c_offset = col_offset_ilen[ilen]
                    worksheet.write(row+r_offset, col+c_offset+0, "-")
                    worksheet.write(row+r_offset, col+c_offset+1, "-")
                    worksheet.write(row+r_offset, col+c_offset+2, "-")
                    worksheet.write(row+r_offset, col+c_offset+3, "-")
                    continue
   
                r_offset = row_offset_nuser[n_user]
                c_offset = col_offset_ilen[ilen]
                worksheet.write(row+r_offset, col+c_offset+0, data[model][dur_ilen_user]["#Req"])
                worksheet.write(row+r_offset, col+c_offset+1, data[model][dur_ilen_user]["E2E"])
                worksheet.write(row+r_offset, col+c_offset+2, data[model][dur_ilen_user]["TTFT"])
                worksheet.write(row+r_offset, col+c_offset+3, data[model][dur_ilen_user]["TPOT"])       
        row+=16              

    workbook.close()
    exit()
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the benchmark result from Serving tests.")
    parser.add_argument(
        "-i",
        type=str,
        help="The benchmark json file",
        required=True,
    )
    parser.add_argument(
        "-o",
        type=str,
        help="out excel filename",
        default="result.xlsx"
    )

    args = parser.parse_args()
    ProcessData(args)

'''
python scripts/GenerateExcel.py -i benchmark_1115.json -o excel/vllm_1115.xlsx


'''