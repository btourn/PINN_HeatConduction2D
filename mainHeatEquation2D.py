
from ImportPackages import *

from PlotClass import PlotClass
from PINN_Module import *
from PINN_DataModule import *
from Callbacks import *
from ActivationAndInitializationFunctions import init_xavier


def train_func(config): 

    trial_dir = ray.train.get_context().get_trial_dir()

    # Generate data
    dataModule = PINN_DataModule(config)

    # Generate model
    model = Backbone(config)
    init_xavier(model)
    PINN_model = PINN_Model(model, config)

    tensorboard = pl_loggers.TensorBoardLogger(save_dir=trial_dir, name="logs") #"./")#, name=config["log_path"])
    csvlogger   = pl_loggers.CSVLogger(save_dir=trial_dir, name="logs") #"./")#, name=config["log_path"])
    printingCallbacks = PrintingCallback()
    metricTracker     = MetricTracker()
    timer             = Timer()
    modelCheckpoints  = ModelCheckpoint(
        monitor='loss_val', 
        mode='min', 
        every_n_epochs=1, 
        save_on_train_epoch_end=True)#, 
        #save_top_k=1)
    callbacks = [printingCallbacks, 
                 metricTracker, 
                 #modelCheckpoints, 
                 timer, 
                 CustomRayTrainReportCallback()]
                 #EarlyStopping(monitor='loss_val', mode='min', verbose=True)]
    trainer = pl.Trainer(
        strategy=RayDDPStrategy(),
        devices="auto",
        accelerator=PINN_model.Device.type,
        max_epochs=model.NumberOfEpochs,
        #default_root_dir=trial_dir, #"./",
        logger=[tensorboard, csvlogger],
        deterministic=True,
        callbacks=callbacks,
        plugins=[RayLightningEnvironment()],
        inference_mode=False,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(PINN_model, datamodule=dataModule)
    
    
    # Make predictions on this trial
    print("##############   Evaluating Model   ##############")
    T_pred_by_batch = trainer.predict(PINN_model, datamodule=dataModule) #, ckpt_path="best")
    print("************Prediccion realizada!!!!!!!")
    T_pred = torch.cat(T_pred_by_batch, axis=0)
    T_exact = dataModule.T_exact
    errors_eval = T_pred - T_exact
    errors_eval_pct = T_pred/T_exact
    #plot.pointwiseValues(T_pred, dataModule, trial_dir, "predicted_temperature.png")
    #plot.pointwiseValues(T_exact, dataModule, trial_dir, "exact_temperature.png")
    #plot.pointwiseValues(errors_eval, dataModule, trial_dir, "point_wise_temp_diff_errors.png")
    #plot.pointwiseValues(errors_eval_pct, dataModule, trial_dir, "point_wise_temp_pct_errors.png")
    print('trial_dir + /T_pred.pt:', trial_dir + '/T_pred.pt')
    asdasddas
    torch.save(T_pred, trial_dir + '/T_pred.pt')
    torch.save(T_exact, trial_dir + '/T_exact.pt')



def tune_PINN_asha(ray_trainer, scheduler, log_path, search_space, num_samples=10):
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="loss_val",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        #run_config=RunConfig(
        #    name=log_path,
        #    storage_path="D:/DOCUMENTOS/EIA-UBA/ProyectoFinal/Codigos/HeatConduction1D/" + log_path,
        #    checkpoint_config=CheckpointConfig(
        #        num_to_keep=1,
        #        checkpoint_frequency=2,
        #    ),
        #)
    )
    return tuner.fit()




def main(inputData):
    
    # Directory path
    data_time_now = datetime.now().strftime("%d%m%Y_%H%M%S")
    suffix = '_experiment'
    log_path = '%s' % (data_time_now) + suffix
    if not os.path.exists("./" + log_path):
        os.makedirs("./" + log_path)

    dir_path = "D:/DOCUMENTOS/EIA-UBA/ProyectoFinal/Codigos/HeatConduction1D/"
    dir_path_full = dir_path + log_path

    # Initialize datasets using an empty configuration
    config0 = {
        "parent_path": dir_path,
        "log_path": dir_path_full,
        "dir_path": None,
    }
    dm0 = PINN_DataModule(config0)
    dm0.prepare_data()
    #plot.plotDistribution(dir_path_full, 'train')

    # Initialize plot class
    plot = PlotClass()

    # Search space for experiment
    search_space = {
        "NonDimensional": tune.grid_search([0, 1]),
        "HardImposedDirichletBC": tune.grid_search([0, 1]),
        "UseLabelledData": tune.grid_search([0, 1]),
        "parent_path": dir_path,
        "log_path": log_path,
        "dir_path": dir_path_full,
    }

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=False, resources_per_worker={"CPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            #num_to_keep=2,
            checkpoint_score_attribute="loss_val",
            checkpoint_score_order="min",
            #checkpoint_frequency=10,
        ),
        local_dir=dir_path_full,
        name="RayResults",
    )
    

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    epochs = inputData["Network properties"]["Epochs"]["Value"]
    scheduler = ASHAScheduler(max_t=epochs, grace_period=100, reduction_factor=2)

    # Train the model with grid search strategy
    results = tune_PINN_asha(ray_trainer, scheduler, dir_path, search_space, num_samples=1)
    
    best_result = results.get_best_result(metric="loss_val", mode="min")
    best_ckpt_path = glob.glob(best_result.checkpoint.path + "/checkpoint.ckpt")[0]
    best_config = {
        "NonDimensional": best_result.config["train_loop_config"]["NonDimensional"],
        "HardImposedDirichletBC": best_result.config["train_loop_config"]["HardImposedDirichletBC"],
        "UseLabelledData": best_result.config["train_loop_config"]["UseLabelledData"],
        "parent_path": dir_path,
        "log_path": log_path,
        "dir_path": dir_path_full,
    }
    outputDict = json.dumps(best_result.__dict__["metrics"])
    outputFile = open(dir_path_full + "/best_result_metrics.json", "w")
    json.dump(outputDict, outputFile, indent=4)
    outputFile.close() 

    best_model = Backbone(best_config)
    init_xavier(best_model)
    best_PINN_model = PINN_Model(best_model, best_config)
    best_dataModule = PINN_DataModule(best_config)
    #plot.trainAndValidationErrors(metricTracker.collectionTrain, 'train', "./" + log_path)
    #plot.trainAndValidationErrors(metricTracker.collectionValidation, 'validation', "./" + log_path)

    tensorboard = pl_loggers.TensorBoardLogger(save_dir="./", name=log_path)
    csvlogger   = pl_loggers.CSVLogger(save_dir="./", name=log_path)
    printingCallbacks = PrintingCallback()
    metricTracker     = MetricTracker()
    timer             = Timer()
    modelCheckpoints  = ModelCheckpoint(
        monitor='loss_val', 
        mode='min', 
        every_n_epochs=1, 
        save_on_train_epoch_end=True, 
        save_top_k=1)
    callbacks = [printingCallbacks, 
                 metricTracker, 
                 #modelCheckpoints, 
                 timer]
                 #EarlyStopping(monitor='loss_val', mode='min', verbose=True)]
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=epochs,
        default_root_dir="./",
        logger=[tensorboard, csvlogger],
        deterministic=True,
        callbacks=callbacks,
        inference_mode=False,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )

    #Testing the model
    print("##############   Testing Model   ##############")
    timer.start_time("test")
    trainer.test(best_PINN_model, datamodule=best_dataModule, ckpt_path=best_ckpt_path)
    elapsed_time_test = timer.time_elapsed("test")
    print("\nTesting Time: ", elapsed_time_test)

    # Make predictions
    print("##############   Evaluating Model   ##############")
    T_pred_by_batch = trainer.predict(best_PINN_model, datamodule=best_dataModule, ckpt_path=best_ckpt_path)
    T_pred = torch.cat(T_pred_by_batch, axis=0)
    T_exact = best_dataModule.T_exact
    errors_eval = T_pred - T_exact
    errors_eval_pct = T_pred/T_exact
    plot.pointwiseValues(T_pred, best_dataModule, "./" + log_path, "predicted_temperature.png")
    plot.pointwiseValues(T_exact, best_dataModule, "./" + log_path, "exact_temperature.png")
    plot.pointwiseValues(errors_eval, best_dataModule, "./" + log_path, "point_wise_temp_diff_errors.png")
    plot.pointwiseValues(errors_eval_pct, best_dataModule, "./" + log_path, "point_wise_temp_pct_errors.png")
    torch.save(T_pred, log_path + '/T_pred.pt')
    torch.save(T_exact, log_path + '/T_exact.pt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile',
                    help="JSON file to be processed",
                    type=argparse.FileType('r'),
                    default="./inputData.json")
    args = parser.parse_args()
    inputData = json.load(args.infile)

    main(inputData)

