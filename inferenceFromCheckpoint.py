from PlotClass import PlotClass
from PINN_Module import *
from PINN_DataModule import *
from Callbacks import *
from ActivationAndInitializationFunctions import init_xavier


def main():
    
    # Check device availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Directory to read data from
    ckpt_path = "05062024_222528_labelledData0_nonDimensional1/version_0/checkpoints/epoch=299-step=600.ckpt"
    log_path = ckpt_path[:ckpt_path.index("/")+1]

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Retrieve structures from checkpoint
    model = ckpt['hyper_parameters']['backbone']
    network_properties = ckpt['hyper_parameters']['network_properties']
    problem_description = ckpt['hyper_parameters']['problem_description']
    
    # Initialize plot class
    plot = PlotClass()

    # Load data module
    # dataModule = torch.load(log_path + '_dataModule.pt', map_location=device)
    # with open(log_path + '_dataModule.pkl', 'rb') as f:
    #     dataModule = pickle.load(f)
    
    # Load model
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

    
    ndf = problem_description["NonDimensionalFactors"]
    kx = ndf["Space"]
    kt = ndf["Time"]
    T_ref = ndf["Temperature"]

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
    epoch = ckpt['epoch']
    ts = 25. # Desired time instant. position of source will be xs=vs*ts
    r0 = 2. # Desired characteristic radius
    xy_data = problem_description["PhysicalDomain"]
    x = torch.linspace(xy_data["LeftCoordinate"], xy_data["RightCoordinate"], 1000)
    y = torch.linspace(xy_data["BottomCoordinate"], xy_data["TopCoordinate"], 1000)
    ms_x, ms_y = torch.meshgrid(x, y)
    x_pred = ms_x.flatten().view(-1, 1)
    y_pred = ms_y.flatten().view(-1, 1)
    t_pred = torch.ones_like(x_pred)*ts*kt 
    r_pred = torch.ones_like(x_pred)*r0*kx
    XY_pred = torch.cat([x_pred, y_pred, t_pred, r_pred], axis=1)
    y_hat = PINN_model.backbone(XY_pred)
    ms_values = y_hat.reshape(ms_x.shape)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection = '3d')
    surf1 = ax.plot_surface(
        ms_x.detach().numpy()/kx, 
        ms_y.detach().numpy()/kx, 
        ms_values.detach().numpy()*T_ref, 
        cmap=cm.jet, 
        linewidth=0, 
        antialiased=False)
    fig.colorbar(surf1, shrink=0.5, aspect=5)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.close(fig)
    #plt.show()
    #fig_name_path = log_path + "inferenceFromChekpoint_epoch" + str(epoch) + ".png"
    #plt.savefig(fig_name_path)


    # Plot several images to make a video
    ti = problem_description['TimeDomain']['InitialTime']/kt
    tf = problem_description['TimeDomain']['FinalTime']/kt
    t_values = np.linspace(ti, tf, 51) # Desired time instant. position of source will be xs=vs*ts
    r0 = 2. # Desired characteristic radius
    xy_data = problem_description["PhysicalDomain"]
    x = torch.linspace(xy_data["LeftCoordinate"], xy_data["RightCoordinate"], 1000)
    y = torch.linspace(xy_data["BottomCoordinate"], xy_data["TopCoordinate"], 1000)
    ms_x, ms_y = torch.meshgrid(x, y)
    x_pred = ms_x.flatten().view(-1, 1)
    y_pred = ms_y.flatten().view(-1, 1)
    r_pred = torch.ones_like(x_pred)*r0*kx

    for i, ts in enumerate(t_values):
        t_pred = torch.ones_like(x_pred)*ts*kt 
        XY_pred = torch.cat([x_pred, y_pred, t_pred, r_pred], axis=1)
        y_hat = PINN_model.backbone(XY_pred)
        ms_values = y_hat.reshape(ms_x.shape)

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection = '3d')
        surf1 = ax.plot_surface(
            ms_x.detach().numpy()/kx, 
            ms_y.detach().numpy()/kx, 
            ms_values.detach().numpy()*T_ref, 
            cmap=cm.jet, 
            linewidth=0, 
            antialiased=False)
        fig.colorbar(surf1, shrink=0.5, aspect=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('T')
        #plt.show()
        num, decimal = str(ts).split('.')
        fig_name_path = log_path + "T_2_"
        plt.savefig(f"{fig_name_path}{num.zfill(3)}.png", bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":

    main()



