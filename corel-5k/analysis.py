import matplotlib  
import matplotlib.pyplot as plt 
import json
import os

def draw_analysis(name, data):
    plt.figure(figsize=(9,9),dpi=400) 
    plt.subplot(221)
    plt.plot(range(len(data['test_recorder']['acc'])),data['test_recorder']['acc']) 
    plt.title('Test Accuracy') 
    plt.subplot(222)
    plt.plot(range(len(data['test_recorder']['loss'])),data['test_recorder']['loss']) 
    plt.title('Test Loss')
    plt.subplot(223) 
    plt.plot(range(len(data['train_recorder']['epoch_acc'])),data['train_recorder']['epoch_acc']) 
    plt.title('Train Accuracy') 
    plt.subplot(224)
    plt.plot(range(len(data['train_recorder']['epoch_loss'])),data['train_recorder']['epoch_loss']) 
    plt.title('Train Loss') 
    # suptitle(name.split('_')[0]+'训练&测试结果图');
    plt.savefig(IMG_PATH+name.split('.')[0]+".jpg") 
    plt.close()

LOG_PATH = './server_source/last_log/'
IMG_PATH = './server_source/analysis/'
for log_file in os.listdir(LOG_PATH):
    if log_file.startswith('.')==False:
        with open(LOG_PATH+log_file) as f:
            data = json.load(f)
            draw_analysis(log_file, data)


