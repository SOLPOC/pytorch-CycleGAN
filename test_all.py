# import os
# def test_all(models):
#       for model in models:
#             print("Start to test "+model)
#             cmd = "python train
#             output_lines = os.popen(cmd).readlines()
#             pattern = r"FID:\s*(\d+\.\d+)"
#             matches = re.findall(pattern, str(output_lines))
#             FID = float(str(matches[0]))