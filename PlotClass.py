from ImportPackages import *

class PlotClass():
    
    def __init__(self, ):
        super().__init__()
        

    def plotDistribution(self, path, stage):
    
        # Plot the locations of the collocation points in the space-time domain        
        if stage=='train':
            title = 'Training points'
        elif stage=='val':
            title = 'Validation points'
        elif stage=='test':
            title = 'Testing points'
        elif stage=='predict':
            title = 'Prediction points'
        else:
            ValueError('Wrong stage. Valid modes are "train", "validate", "test", or "predict"!')

        X = torch.load(path + '/' + stage + '.pt')
        if stage=='predict':
            X_pred = X[:][0]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(X_pred[:, 1:2].data.cpu().numpy(),
                    X_pred[:, 0:1].data.cpu().numpy(), 
                    c='r', s=1)
        else:
            X_dom  = X["Domain"][:][0]
            X_bc   = X["BoundaryCondition"][:][0]
            X_ic   = X["InitialCondition"][:][0]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(X_dom[:, 1:2].data.cpu().numpy(),
                    X_dom[:, 0:1].data.cpu().numpy(), 
                    c='r', s=1, label='Domain')
            ax.scatter(X_bc[:, 1:2].data.cpu().numpy(),
                    X_bc[:, 0:1].data.cpu().numpy(), 
                    c='b', s=1, label='BC')
            ax.scatter(X_ic[:, 1:2].data.cpu().numpy(),
                    X_ic[:, 0:1].data.cpu().numpy(), 
                    c='g', s=1, label='IC')
            if "LabelledData" in X.keys():
                X_data = X["LabelledData"][:][0]
                ax.scatter(X_data[:, 1:2].data.cpu().numpy(),
                        X_data[:, 0:1].data.cpu().numpy(), 
                        c='m', s=1, label='LabelledData')

        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel('Time domain (t)')
        plt.ylabel('Space domain (x)')
        #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_title(title)
        fig_name_path = path + "/collocation_points_" + stage + ".png"
        fig.savefig(fig_name_path, bbox_extra_artists=(lgd,), bbox_inches='tight')


    def trainAndValidationErrors(self, errorsDict, stage, path):
        n_epochs = len(errorsDict)
        cols = len(errorsDict[0])
        errors_keys = errorsDict[0].keys()
        fig, ax = plt.subplots(1, cols, figsize=(15, 3))
        colors = ['black', 'red', 'blue', 'green']
        titles = ['Overall Loss', 'PDE Loss', 'BC Loss', 'IC Loss']
        if cols==5:
            colors.append('magenta')
            titles.append('Data Loss')
            
        for i, keys in enumerate(errors_keys):
            errors = []
            for j in range(n_epochs):
                errors.append(errorsDict[j][keys])

            ax[i].loglog(np.arange(0, n_epochs), errors, color=colors[i])
            ax[i].set_xlabel('Epoch')
            ax[i].set_title(titles[i])
        
        fig_name_path = path + "/" + stage + "_errors.png"
        plt.savefig(fig_name_path)
        
    def pointwiseValues(self, values, dataModule, path, fig_name):
        x_test = (dataModule.x_test).detach().numpy()
        t_test = (dataModule.t_test).detach().numpy()
        ms_x, ms_t = np.meshgrid(x_test, t_test)
        ms_values = values.reshape(ms_x.shape).detach().numpy()
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection = '3d')
        #ax.view_init(elev=20, azim=-90)
        surf1 = ax.plot_surface(ms_x, ms_t, np.transpose(ms_values), cmap=cm.jet, linewidth=0, antialiased=False)
        fig.colorbar(surf1, shrink=0.5, aspect=5)
        plt.xlabel('x')
        plt.ylabel('t')
        fig_name_path = path + "/" + fig_name
        plt.savefig(fig_name_path)
        
        
        