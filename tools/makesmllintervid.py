# import os
# import json
# from tqdm import tqdm
# # 设置视频文件夹和 JSONL 文件路径
# video_dir = '/home/lijunjie/work/datasets/InternVId-FLT/InternVId-FLT_1'
# jsonl_file = '/home/lijunjie/work/datasets/InternVId-FLT/InternVid-10M-flt.jsonl'  
# output_jsonl = '/home/lijunjie/work/datasets/InternVId-FLT/InternVId-FLT_1.jsonl'  # 新 JSONL 文件的路径

# # 收集所有视频文件名（去掉扩展名）
# video_files = {os.path.splitext(file)[0] for file in os.listdir(video_dir) if file.endswith('.mp4')}
# cnt = 0
# # 读取和写入 JSONL 文件，包含进度条
# with open(jsonl_file, 'r') as infile, open(output_jsonl, 'w') as outfile:
#     lines = infile.readlines()
#     for line in tqdm(lines, desc="Filtering JSONL", unit="line"):  # tqdm 添加进度条
#         data = json.loads(line)
#         video_name = data['YoutubeID'] + "_" + data['Start_timestamp'] + "_" + data["End_timestamp"]  # 假设 JSONL 文件中使用 'YoutubeID' 作为视频名称键
#         if video_name in video_files:
#             cnt += 1
#             outfile.write(line)  # 如果视频存在，则将该行写入新的 JSONL 文件
# print(cnt)
# print("Filtered JSONL file created successfully.")


import os
breakpoint()
os.path.exists("/home/lijunjie/work/datasets/InternVId-FLT/InternVId-FLT_1/43HEVO8tx6s_00:04:34.467_00:04:35.367.mp4")