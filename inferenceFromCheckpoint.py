from PlotClass import PlotClass
from PINN_Module import *
from PINN_DataModule import *
from Callbacks import *
from ActivationAndInitializationFunctions import init_xavier


def main():
    
    # Directory to read data from
    ckpt_path = "23052024_114506_labelledData0_nonDimensional1/version_0/checkpoints/epoch=299-step=300.ckpt"
    print("Checkpoint path: ", ckpt_path)
    log_path = ckpt_path[:ckpt_path.index("/")+1]
    start, end = "=", "-"
    epoch = ckpt_path[ckpt_path.find(start)+len(start):ckpt_path.rfind(end)]

    # Load pickle variables from directory
    with open(log_path + '_problem_description.pkl', 'rb') as f:
        problem_description = pickle.load(f)
    with open(log_path + '_collocation_points.pkl', 'rb') as f:
        collocation_points = pickle.load(f)
    with open(log_path + '_network_properties.pkl', 'rb') as f:
        network_properties = pickle.load(f)

    
    # Check device availability
    device_type = network_properties["Device"]
    if device_type == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            device_type = 'cpu'
    elif device_type == 'gpu':
        device = torch.device('gpu') if torch.cuda.is_available() else torch.device("cpu")
    elif device_type == 'cpu':
        device = torch.device('cpu')
    network_properties["Device"] = device
            
    
    model = Backbone(network_properties, problem_description)
    input_dict = {
        "backbone": model, 
        "network_properties": network_properties, 
        "problem_description": problem_description
        }
    PINN_model = PINN_Model.load_from_checkpoint(
        checkpoint_path=ckpt_path, 
        map_location=device,
        **input_dict
        )


    '''PINN_model.train()
    ds_test = torch.load(log_path + "test.pt")
    XY_test = ds_test["Domain"][:][0]
    T_test  = ds_test["Domain"][:][1]
    XY_test.requires_grad = True
    T_hat = PINN_model.backbone(XY_test)
    res = PINN_model.governingEquationsResidue(XY_test, T_hat)'''


    # disable randomness, dropout, etc...
    PINN_model.eval()

    # predict with the model
    xy_data = problem_description["PhysicalDomain"]
    x = torch.linspace(xy_data["LeftCoordinate"], xy_data["RightCoordinate"], 1000)
    y = torch.linspace(xy_data["BottomCoordinate"], xy_data["TopCoordinate"], 1000)
    ms_x, ms_y = torch.meshgrid(x, y)
    x_pred = ms_x.flatten().view(-1, 1)
    y_pred = ms_y.flatten().view(-1, 1)
    t_pred = torch.ones_like(x_pred)*25 #0.01
    r_pred = torch.ones_like(x_pred)*4 #0.01
    XY_pred = torch.cat([x_pred, y_pred, t_pred, r_pred], axis=1)
    y_hat = PINN_model.backbone(XY_pred)
    ms_values = y_hat.reshape(ms_x.shape)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection = '3d')
    surf1 = ax.plot_surface(
        ms_x.detach().numpy(), 
        ms_y.detach().numpy(), 
        np.transpose(ms_values.detach().numpy()), 
        cmap=cm.jet, 
        linewidth=0, 
        antialiased=False)
    fig.colorbar(surf1, shrink=0.5, aspect=5)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()
    fig_name_path = log_path + "inferenceFromChekpoint_epoch" + epoch + ".png"
    plt.savefig(fig_name_path)
    print("End")


if __name__ == "__main__":

    main()



