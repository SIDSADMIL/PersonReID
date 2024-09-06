import torch
import os

def savemodelfile(Dir,Name,Model,Optimizer,Checkpoint):

    # Load the checkpoint
    checkpoint = torch.load(Checkpoint)
    # Extract the state_dict
    state_dict = checkpoint['state_dict']
    o_st_dt=checkpoint['optimizer']#['state']
    Model.load_state_dict(state_dict)
    Optimizer.load_state_dict(o_st_dt)
 
    if not os.path.exists(Dir):
        os.mkdir(Dir)
    # Save the model state
    print('model keys :',Model.state_dict().keys())
    torch.save({'state_dict': Model.state_dict(),
                'optimizer': Optimizer.state_dict()},
               Dir + "/"+Name+".pth")
