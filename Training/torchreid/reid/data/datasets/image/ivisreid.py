import os
import os.path as osp
from torchreid.reid.data.datasets.dataset import ImageDataset
import torchreid

class iVISREid(ImageDataset):

    #dataset_dir = 'iVISREid_REid_Dataset'

    def __init__(self, root='', **kwargs):
        # constructor of the class, here define all required paths and call the parent implementation
        # by calling the parent implementation, no need to override the methods as the parent method will do the same
        
        self.root = osp.abspath(osp.expanduser(root))
        
        self.dataset_dir = self.root#os.path.join(self.root, self.dataset_dir)
        
        train = os.path.join(self.dataset_dir, "Train")
        query = os.path.join(self.dataset_dir, "Query")
        gallery = os.path.join(self.dataset_dir, "Gallery")
        
        self.train = self.generate_data_list(train)
        self.query = self.generate_data_list(query)
        self.gallery = self.generate_data_list(gallery)
        # call parent implementation
        super(iVISREid, self).__init__(self.train, self.query, self.gallery, **kwargs)
        #torchreid.reid.data.register_image_dataset('testdata_set', NewDataset)

    def generate_data_list(self, rootfolder):
        if isinstance(rootfolder, list):
            # Assuming it's a list of paths, take the first path
            rootfolder = rootfolder[0]
        if isinstance(rootfolder, tuple):
            # Extract the path from the tuple
            rootfolder = rootfolder[0]
        data_list = []
        # Check if folder exists and is a directory
        if os.path.isdir(rootfolder):
            print("Root Folder To Extract Data: ", rootfolder)
            for Pfolder in os.listdir(rootfolder):
                Pfolderpath=os.path.join(rootfolder, Pfolder)
                Pid = Pfolder.split('Person')[1]
                for Cfolder in os.listdir(Pfolderpath):
                    Cfolderpath=os.path.join(Pfolderpath, Cfolder)
                    Cid = Cfolder.split('Camera')[1]
                    for Ifile in os.listdir(Cfolderpath):
                        if Ifile.endswith('.jpg'):
                            Ifilepath = os.path.join(Cfolderpath, Ifile)
                            data_list.append((Ifilepath, Pid, Cid))
        else:
            print(f"Invalid directory: {rootfolder}")
        
        print("Total images: ", len(data_list))
        return data_list