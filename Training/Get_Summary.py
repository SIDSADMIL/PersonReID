import datetime

def get_summary(logfilepath):
    inputfile=logfilepath
    dir_split = (logfilepath.split('/'))[:-1]
    delim = '/'
    dir_path = delim.join(folder for folder in dir_split)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Create a file name with the timestamp
    filename = f"TrainingSummary_{timestamp}.txt"
    outputfile=dir_path+'/'+filename
    file = open(inputfile,'r')
    Ofile = open(outputfile,'a')
    content = file.read()
    epochs = []
    mAPs = []
    str_split = content.split('Epoch                     ')#'Epoch
    #print(str_split[0][0:5])
    for i in range(len(str_split)):
        if i != 0:
            epoch = str_split[i].split('\n_')
            epochs.append(epoch[0])
            print(epoch[0])
            mAP_split = str_split[i].split('mAp :')
            #mAP_Percent = mAP_split[1].split('\n_')
            #mAP = mAP_Percent[0].split('%')[0]
            mAP = mAP_split[1].split('\n')[0].strip()
            mAPs.append(mAP)
            print(mAP)
            Ofile.write(str(f'Epoch: {epoch[0]}, mAP: {mAP}\n'))

    for i in range(len(epochs)):
        Ofile.write(str(f'Epoch: {epochs[i]}, mAP: {mAPs[i]}\n'))
    file.close()
    Ofile.close()
'''
fp = 'C:/IVIS/PersonReID/Training/log/training1/IVIS_Coustom_Dataset/2024_08_29_00_08_59/TrainLog.txt'
get_summary(fp)
'''