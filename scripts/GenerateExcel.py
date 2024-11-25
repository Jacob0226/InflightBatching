import argparse
import json
import os
import sys
import pandas as pd
import xlsxwriter
import os

def FillInExcelTemplate(row, col, model_name, worksheet):
    worksheet.write(row, col, model_name)
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
    worksheet.write(row+3, col, "1")
    worksheet.write(row+4, col, "32")
    worksheet.write(row+5, col, "64")


def ProcessData(args):
    workbook = xlsxwriter.Workbook(args.o)
    worksheet = workbook.add_worksheet()

    Models = ["Llama-3.1-8B", "Llama-3.1-70B"]
    TP= [4, 2]
    Dtype = ["float16"] # ["float16", "float8"]

    Dur="2m"
    I_Len=["i2500", "i5500", "i11000"]
    N_User=[1, 32, 64]

    # Excel (x,y) position
    row, col = 0, 0
    row_offset_nuser={1:3, 32:4, 64:5}
    col_offset_ilen={"i2500":1, "i5500":5, "i11000":9}

    with open(args.i, 'r') as f:
        data = json.load(f)  # Load JSON data as a Python dictionary
    if data['vLLM'] == {}: # Empty dictionary
        data = data['Triton']
    elif data['Triton'] == {}: # Empty dictionary
        data = data["vLLM"]
    
    for tp in TP:
        for model in Models:
            for dtype in Dtype:
                model_name = '_'.join([model, dtype, 'TP'+str(tp)])
                print(model_name)
                if model_name not in data:
                    print(f"model {model_name} not in json. Skip...")
                    continue
                # Fill in the excel table 
                FillInExcelTemplate(row, col, model_name, worksheet)

                for ilen in I_Len:
                    for n_user in N_User:
                        n_user_str = "{:02}".format(n_user) + "user"
                        dur_ilen_user = '/'.join(['2m', ilen, n_user_str])

                        # print(f"dur_ilen_user={dur_ilen_user}")
                        # print(f"model_name={model_name}")
                        # print(f"data[model_name][dur_ilen_user]={data[model_name][dur_ilen_user]}")
                        # exit(0)

                        
                        r_offset = row_offset_nuser[n_user]
                        c_offset = col_offset_ilen[ilen]
                        worksheet.write(row+r_offset, col+c_offset+0, data[model_name][dur_ilen_user]["#Req"])
                        worksheet.write(row+r_offset, col+c_offset+1, data[model_name][dur_ilen_user]["E2E"])
                        worksheet.write(row+r_offset, col+c_offset+2, data[model_name][dur_ilen_user]["TTFT"])
                        worksheet.write(row+r_offset, col+c_offset+3, data[model_name][dur_ilen_user]["TPOT"])       
                row+=8              

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