import gdown

url = 'https://drive.google.com/u/0/uc?id=1-qjJVc1T9IRIqaxHKdRsAYw4-2v-hBuN&export=download'
output = "./model_NLG.ckpt"
gdown.download(url, output)

url = 'https://drive.google.com/u/0/uc?id=1v8NoP-wdf9QVfMgGqccvw9-A7jnlyke0&export=download'
output = "./model_classifer_new.ckpt"
gdown.download(url, output)