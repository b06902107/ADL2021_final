import gdown
import zipfile

url = "https://drive.google.com/u/1/uc?id=1BNKD9Hbvt1q6fyVxCOqTVLI-GpRK_Cq3&export=download"
output = "./dst_models.zip"
gdown.download(url, output)

with zipfile.ZipFile("./dst_models.zip", 'r') as zip_ref:
    zip_ref.extractall('.')
