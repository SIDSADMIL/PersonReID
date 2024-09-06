import os
import os.path as osp
from torchreid.reid.data.datasets.dataset import ImageDataset
import torchreid

class REid(ImageDataset):

    #dataset_dir = 'REid_Dataset'

    def __init__(self, root='', **kwargs):
        # constructor of the class, here define all required paths and call the parent implementation
        # by calling the parent implementation, no need to override the methods as the parent method will do the same

        self.root = osp.abspath(osp.expanduser(root))

        self.dataset_dir = self.root#os.path.join(self.root, self.dataset_dir)

        train = os.path.join(self.dataset_dir, "bounding_box_train")
        query = os.path.join(self.dataset_dir, "query")
        gallery = os.path.join(self.dataset_dir, "bounding_box_test")

        self.train = self.generate_data_list(train)
        self.query = self.generate_data_list(query)
        self.gallery = self.generate_data_list(gallery)
        # call parent implementation
        super(REid, self).__init__(self.train, self.query, self.gallery, **kwargs)
        #torchreid.reid.data.register_image_dataset('testdata_set', NewDataset)

    def generate_data_list(self, folder):
        if isinstance(folder, list):
            # Assuming it's a list of paths, take the first path
            folder = folder[0]

        if isinstance(folder, tuple):
            # Extract the path from the tuple
            folder = folder[0]
        data_list = []
        person_id = 0
        camera_id = 0
        # Check if folder exists and is a directory
        if os.path.isdir(folder):

            for imgfile in os.listdir(folder):
                if imgfile.endswith('.jpg'):
                    #0001_c1s1_001051_00
                    #try:

                    splits = str(imgfile).split('_c')

                    person_id = int(splits[0])  # Extract person ID from folder name
                    splits = str(splits[1]).split('s')

                    camera_id = int(splits[0])  # Extract camera ID from folder name
                    camera_id -= 1
                    #except Exception as e:
                       # print("")
                    #else:
                       # person_id=0
                       # camera_id=1
                    image_path = os.path.join(folder, imgfile)
                    # this framework uses a format of (Path, PID, CID) tuple
                    # So we have to build / extract the data accordingly
                    data_list.append((image_path, person_id, camera_id))
        else:
            print(f"Invalid directory: {folder}")

        print("Total images: ", len(data_list))
        return data_list