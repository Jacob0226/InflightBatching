import argparse
import json
import os
import sys
import pandas as pd
import xlsxwriter
import os

'''
The structure of input folder:
    report_1106/
    ├── Llama-3.1-70B_float16_TP2
    │   └── 2m
    │       ├── i11000
    │       │   ├── 01user : contain benchmark_00.json
    │       │   ├── 32user : contain benchmark_00~31.json
    │       │   └── 64user : contain benchmark_00~63.json
    │       ├── i2500
    │       │   ├── 01user : same as above...
    │       │   ├── 32user
    │       │   └── 64user
    │       └── i5500
    │           ├── 01user
    │           ├── 32user
    │           └── 64user
    ├── Llama-3.1-70B_float16_TP4
    │  ...
'''

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
    workbook = xlsxwriter.Workbook(args.out)
    worksheet = workbook.add_worksheet()

    root_folder = args.i_folder
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
    
    
    for tp in TP:
        for model in Models:
            for dtype in Dtype:
                model_name = '_'.join([model, dtype, 'TP'+str(tp)])
                # Fill in the excel table 
                FillInExcelTemplate(row, col, model_name, worksheet)

                for ilen in I_Len:
                    for n_user in N_User:
                        n_user_str = "{:02}".format(n_user) + "user"
                        json_folder = os.path.join(root_folder, model_name, Dur, ilen, n_user_str)
                        if os.path.exists(json_folder)==False:
                            print(f"json_folder={json_folder} is not exists.")
                        else:
                            # Calculate the #Req, E2E, TTFT and TPOT
                            print(f"Prcossing {json_folder}")
                            n_completed_req, E2E, TTFT, TPOT = 0, [], [], []
                            avg_E2E, avg_TTFT, avg_TPOT = "None", "None", "None"
                            for id in range(n_user):
                                id_str = "{:02}".format(id)
                                json_file = os.path.join(json_folder, "benchmark_" + id_str + ".json")
                                if os.path.exists(json_file)==False:
                                    print(f"Error: json_folder={json_folder} doesn't have exact {n_user} json files,")
                                    print(f"  E.g, lack of [{json_file}], save benchmark as None")
                                    break

                                # Open the JSON file and load the data
                                with open(json_file, 'r') as file:
                                    data = json.load(file)
                                n_completed_req += len(data["E2E"]) # Not supported yet
                                E2E += data["E2E"]
                                TTFT += data["TTFT"]
                                TPOT += data["TPOT"]

                            # Write metric into excel
                            if n_completed_req > 0:
                                avg_E2E = sum(E2E) / n_completed_req
                                avg_TTFT = sum(TTFT) / n_completed_req
                                avg_TPOT = sum(TPOT) / n_completed_req
                                # print(f"#Req = {n_completed_req}, avg_E2E = {avg_E2E}, avg_TTFT = {avg_TTFT}, avg_TPOT = {avg_TPOT}")
                            else:
                                n_completed_req = "None"
                            r_offset = row_offset_nuser[n_user]
                            c_offset = col_offset_ilen[ilen]
                            worksheet.write(row+r_offset, col+c_offset+0, n_completed_req)
                            worksheet.write(row+r_offset, col+c_offset+1, avg_E2E)
                            worksheet.write(row+r_offset, col+c_offset+2, avg_TTFT)
                            worksheet.write(row+r_offset, col+c_offset+3, avg_TPOT)       
                row+=8              

    workbook.close()
    exit()
                    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the benchmark result from Serving tests.")
    parser.add_argument(
        "--i-folder",
        "-i",
        type=str,
        help="Specify the folder which contains the the serving results",
        required=True,
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        help="out excel filename",
        default="result.xlsx"
    )

    args = parser.parse_args()
    ProcessData(args)

'''
python scripts/GenerateExcel.py -i report_1107 -o TRTLLM_1107.xlsx


'''