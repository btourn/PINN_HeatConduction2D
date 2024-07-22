
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
    ckpt_path = "05062024_222528_labelledData0_nonDimensional1/version_0/checkpoints/epoch=899-step=1800.ckpt"
    log_path = ckpt_path[:ckpt_path.index("/")+1]

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Retrieve structures from checkpoint
    model = ckpt['hyper_parameters']['backbone']
    network_properties = ckpt['hyper_parameters']['network_properties']
    problem_description = ckpt['hyper_parameters']['problem_description']


    # Load pickle variables from directory
    '''with open(log_path + '_problem_description.pkl', 'rb') as f:
        problem_description = pickle.load(f)
    with open(log_path + '_collocation_points.pkl', 'rb') as f:
        collocation_points = pickle.load(f)
    with open(log_path + '_network_properties.pkl', 'rb') as f:
        network_properties = pickle.load(f)'''    
    
    # Initialize plot class
    plot = PlotClass()

    # Load data module
    try:
        with open(log_path + '_dataModule.pkl', 'rb') as f:
            dataModule = pickle.load(f)
    except:
        collocation_points = ckpt['datamodule_hyper_parameters']['collocation_points']
        labelled_data_points = ckpt['datamodule_hyper_parameters']['labelled_data_points']
        log_path = ckpt['datamodule_hyper_parameters']['log_path']
        dir_path = log_path
        dataModule = PINN_DataModule(problem_description, collocation_points, labelled_data_points, network_properties, log_path, dir_path)
    
    
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


    PINN_model.plotModelParameters()
    X = dataModule.domain('Sobol', 'train', 'Domain')
    PINN_model.plotActivationFunctionsWhileRunning(X)


    # Load trainer
    trainer = torch.load(log_path + '/' + '_trainer.pt', map_location=device)
    with open(log_path + '_trainer.pkl', 'rb') as f:
        trainer = pickle.load(f)
    #trainer.fit_loop.max_epochs = 15

    tensorboard = trainer.loggers[0]
    tensorboard._version = 0
    csvlogger = trainer.loggers[1]
    csvlogger._version = 1
    subdir = '/version_1'
    metrics_fname = 'metrics2.csv'
    csvlogger.experiment.log_dir = './' + log_path + subdir
    csvlogger.experiment.NAME_METRICS_FILE = metrics_fname
    csvlogger.experiment.metrics_file_path = './' + log_path + subdir + '/' + metrics_fname

    for callback in trainer.callbacks:
        if callback.state_key=='Timer':
            timer = callback
    
    # Customize seed
    s = 123
    pl.seed_everything(s)

    # Resume training
    print("##############   Resume Fitting Model   ##############")
    dataModule.DirPath = dataModule.LogPath
    timer.start_time("train")
    trainer.fit(PINN_model, datamodule=dataModule, ckpt_path=ckpt_path)
    elapsed_time_train = timer.time_elapsed("train")
    previous_time = ckpt['callbacks']['Timer']['time_elapsed']['train']
    print("\nTraining time since resume: ", elapsed_time_train)
    print("\nTotal raining time: ", elapsed_time_train + previous_time)
    ckpt_path_opt = glob.glob("./" + log_path + "/version_0/checkpoints/" + "*.ckpt")[0]
    
    
    # Make predictions
    keys, preds = [], []
    ndf = problem_description["NonDimensionalFactors"]
    for key in dataModule.AllKeysPredict:
        dataModule.PredictKey = key
        T_pred = trainer.predict(PINN_model, datamodule=dataModule, ckpt_path='best')[0]
        XY = dataModule.predict
        file = 'predict_' + key
        keys.append(file)
        preds.append([XY, T_pred])
        plot.temperaturePrediction(XY, T_pred, ndf, log_path, file)

    predictions = dict(zip(keys, preds))
    torch.save(predictions, "./" + log_path + "/" + '_predictions.pt')
    #with open("./" + log_path + "/" + '_predictions.pkl', 'wb') as f:
    #    pickle.dump(predictions, f)




if __name__ == "__main__":

    main()
