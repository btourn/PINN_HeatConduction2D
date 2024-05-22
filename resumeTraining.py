
from PlotClass import PlotClass
from PINN_Module import *
from PINN_DataModule import *
from Callbacks import *
from ActivationAndInitializationFunctions import init_xavier


def main():
    
    # Directory to read data from
    ckpt_path = "16052024_193720_labelledData0_nonDimensional1/version_0/checkpoints/epoch=21-step=22.ckpt"
    log_path = ckpt_path[:ckpt_path.index("/")+1]

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
    
    # Initialize plot class
    plot = PlotClass()

    # Load data module
    with open(log_path + '_dataModule.pkl', 'rb') as f:
        dataModule = pickle.load(f)
    
    # Load model
    with open(log_path + '_model.pkl', 'rb') as f:
        model = pickle.load(f)
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

    # Load trainer
    with open(log_path + '_trainer.pkl', 'rb') as f:
        trainer = pickle.load(f)
    tensorboard = trainer.loggers[0]
    tensorboard._version = 0
    csvlogger = trainer.loggers[1]
    csvlogger._version = 1
    csvlogger.experiment.NAME_METRICS_FILE='metrics2.csv'
    
    # Resume training
    print("##############   Resume Fitting Model   ##############")
    trainer.fit(PINN_model, datamodule=dataModule, ckpt_path=ckpt_path)
    ckpt_path_opt = glob.glob("./" + log_path + "/version_0/checkpoints/" + "*.ckpt")[0]
    
    #Testing the model
    print("##############   Testing Model   ##############")
    trainer.test(PINN_model, datamodule=dataModule, ckpt_path=ckpt_path_opt)

    # Make predictions
    print("##############   Evaluating Model   ##############")
    T_pred_by_batch = trainer.predict(PINN_model, datamodule=dataModule, ckpt_path=ckpt_path_opt)
    T_pred = torch.cat(T_pred_by_batch, axis=0)
    T_exact = dataModule.T_exact
    errors_eval = T_pred - T_exact
    errors_eval_pct = T_pred/T_exact
    plot.pointwiseValues(T_pred, dataModule, "./" + log_path, "predicted_temperature.png")
    plot.pointwiseValues(T_exact, dataModule, "./" + log_path, "exact_temperature.png")
    plot.pointwiseValues(errors_eval, dataModule, "./" + log_path, "point_wise_temp_diff_errors.png")
    plot.pointwiseValues(errors_eval_pct, dataModule, "./" + log_path, "point_wise_temp_pct_errors.png")



if __name__ == "__main__":

    main()
