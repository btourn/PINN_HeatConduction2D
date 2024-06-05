
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
    ckpt_path = "04062024_152456_labelledData0_nonDimensional1/version_0/checkpoints/epoch=199-step=200.ckpt"
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
    with open(log_path + '_dataModule.pkl', 'rb') as f:
        dataModule = pickle.load(f)
    
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

    # Load trainer
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
    
    #Testing the model
    print("##############   Testing Model   ##############")
    trainer.test(PINN_model, datamodule=dataModule, ckpt_path=ckpt_path_opt)

    # Make predictions
    print("##############   Evaluating Model   ##############")
    timer.start_time("test")
    trainer.test(PINN_model, datamodule=dataModule, ckpt_path=ckpt_path)
    elapsed_time_test = timer.time_elapsed("test")
    print("\nTesting Time: ", elapsed_time_test)

    # Make predictions
    print("##############   Evaluating Model   ##############")
    parentDir = './ExactSolutions'
    files = os.listdir(parentDir)
    keys, preds = [], []
    dataModule.DirPath = log_path
    for file in files:
        if 'ds' in file:
            data = scipy.io.loadmat(parentDir + '/' + file)
            XY = torch.from_numpy(data['XY']).type(torch.FloatTensor)
            T  = torch.from_numpy(data['T']).type(torch.FloatTensor)
            ds = CustomDataset(XY, T)
            dl = DataLoader(dataset=ds, batch_size=len(ds), shuffle=False)
            T_pred_by_batch = trainer.predict(PINN_model, dataloaders=dl, ckpt_path=ckpt_path)
            T_pred = torch.cat(T_pred_by_batch, axis=0)
            fig_name = 'predict' + file[2:-4]
            plot.temperaturePrediction(XY, T_pred, log_path, fig_name)
            keys.append(file)
            preds.append(T_pred)

    predictions = dict(zip(keys, preds))
    with open("./" + log_path + "/" + '_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)




if __name__ == "__main__":

    main()
