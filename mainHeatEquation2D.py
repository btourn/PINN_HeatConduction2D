
from ImportPackages import *

from PlotClass import PlotClass
from PINN_Module import *
from PINN_DataModule import *
from Callbacks import *
from ActivationAndInitializationFunctions import init_xavier


def initialize_inputs():
    
    # Retrieve data from input data file
    rho = 7.6e-6 # Density (kg/mm^3)
    k = 0.025 # Thermal Conductivity (W/mm.K)
    cp = 685.0 # Specific Heat Capacity (J/kg.K)
    alpha = k/(rho*cp) # Thermal Diffusivity
    T0 = 273.0 # Initial temperature (K)
    referenceTemperature = 273.0 # Temperature (K)
    L, H = 100.0, 50.0 # Size of rectangular domain (mm)
    r0_i, r0_f = 1.0, 10.0 # Characteristic radius (mm)
    xi, xf = 0.0, L # x-limits
    yi, yf = -H/2, H/2 # y-limits
    ti, tf = 0.0, 50.0 # Time domain(s)
    vs = 2. #2 # Heat source velocity (mm/s)
    P = 62.83185 # Heat source total power (W)
    ti_s, tf_s = 0.0, 50.0 # Initial and final time of heat source
    x0_s, y0_s = 0.0, 0.0 # Initial position of heat source (mm)
    rho_ref = k_ref = cp_ref = alpha_ref = T_ref = vs_ref = P_ref = kx = kt = 1
    nonDimensional = True
    hardImposedDirichletBC = False
    if nonDimensional:
        rho_ref, k_ref, cp_ref, alpha_ref, T_ref = rho, k, cp, alpha, referenceTemperature
        vs_ref = alpha/L
        P_ref = k*T_ref
        kx, kt = 1/xf, k_ref/(rho_ref*cp_ref*(xf-xi)**2)

    nonDimensionalFactors = {'Space': kx,
                             'Time': kt,
                             'Temperature': T_ref}
    
    dir_path_ = None
    loadDatasetFromDirectory = 0 
    if loadDatasetFromDirectory:
        dir_path_ = ""
            
    # Define data structures 
    material_properites_ = {
        "Dict_Name": "Material properties",
        "Density": rho/rho_ref, 
        "ThermalConductivity": k/k_ref, 
        "SpecificHeatCapacity": cp/cp_ref, 
        "ThermalDiffusivity": alpha/alpha_ref,
        "ReferenceTemperature": referenceTemperature/T_ref # K
    }
    physical_domain_ = {
        "Dict_Name": "Physical domain",
        "LeftCoordinate": xi*kx, # meters
        "RightCoordinate": xf*kx, # meters
        "BottomCoordinate": yi*kx, # meters
        "TopCoordinate": yf*kx, # meters
    }
    time_domain_ = {
        "Dict_Name": "Time domain",
        "InitialTime": ti*kt, # seconds
        "FinalTime": tf*kt # seconds
    }
    ib_conditions_ = {
        "Dict_Name": "Initial and boundary conditions",
        "InitialCondition": T0/T_ref, # K
        "LeftBoundaryCondition": {
            'Type': "Neumann",
            'Value': 0
            },
        "RightBoundaryCondition": {
            'Type': "Neumann",
            'Value': 0
            },
        "BottomBoundaryCondition": {
            'Type': "Neumann",
            'Value': 0 
            },
        "TopBoundaryCondition": {
            'Type': "Neumann",
            'Value': 0 
            }
    }
    heat_source_ = {
        "Dict_Name": "Heat source data",
        "Velocity": vs/vs_ref,
        "TotalPower": P/P_ref,
        "InitialXPosition": x0_s*kx,
        "InitialYPosition": y0_s*kx,
        "InitialTime": ti_s*kt,
        "FinalTime": tf_s*kt,
        "LowerCharacteristicRadius": r0_i*kx,
        "UpperCharacteristicRadius": r0_f*kx,
    }
    lb = [
        float(physical_domain_["LeftCoordinate"]), 
        float(physical_domain_["BottomCoordinate"]), 
        float(time_domain_["InitialTime"]), 
        float(heat_source_["LowerCharacteristicRadius"])]
    ub = [
        float(physical_domain_["RightCoordinate"]), 
        float(physical_domain_["TopCoordinate"]), 
        float(time_domain_["FinalTime"]), 
        float(heat_source_["UpperCharacteristicRadius"])]
    problem_description_ = {
        "Dict_Name": "Problem description",
        "MaterialProperties": material_properites_,
        "PhysicalDomain": physical_domain_,
        "TimeDomain": time_domain_,
        "VariableLowerBounds": lb,
        "VariableUpperBounds": ub,
        "InitialAndBoundaryConditions": ib_conditions_,
        "HeatSource": heat_source_,
        "NonDimensional": nonDimensional,
        "NonDimensionalFactors": nonDimensionalFactors,
        "HardImposedDirichletBC": hardImposedDirichletBC,
    }
    collocation_points_ = {
        "Dict_Name": "Collocation points",
        "Domain": 40000,
        "BoundaryCondition": 2500, 
        "InitialCondition": 10000, 
        "ProportionOfEntriesWithinDisk": 0.2,
        "RadiusOfDisk": 1.*kx
    }
    labelled_data_points_ = {
        "Dict_Name": "Labelled data points",
        "UseLabelledData": 0,
        "NumberOfDataPoints": 1000
    }
    network_properties_ = {
        "Dict_Name": "Network properties",
        "InputDimensions": 4,
        "OutputDimensions": 1,
        "HiddenLayers": 6,
        "NumberOfNeurons": 150,
        "DatasetPartitions": [1., 0., 0.],
        "WeightDecay": 0,
        "Epochs": 15000,
        "LearningRate": 0.8,
        "Activation": "tanh", #"tanh", etc
        "Optimizer": "LBFGS", #"ADAM",
        "Criterion": "MSE",
        "Device": "cpu",
        "BatchSizeTrain": -1,
        "BatchSizeValidation": -1,
        "BatchSizeTest": -1,
        "BatchSizePredict": -1
    }
    
    # Define directory for storing the results
    data_time_now = datetime.now().strftime("%d%m%Y_%H%M%S")
    labelledData = labelled_data_points_["UseLabelledData"]
    suffix = '_labelledData' + str(int(labelledData)) + '_nonDimensional' + str(int(nonDimensional)) 
    log_path_ = '%s' % (data_time_now) + suffix
    if not os.path.exists("./" + log_path_):
        os.makedirs("./" + log_path_)

    # Save variables as pickle
    torch.save(problem_description_, "./" + log_path_ + "/" + '_problem_description.pt')
    torch.save(collocation_points_, "./" + log_path_ + "/" + '_labelled_data_points.pt')
    torch.save(labelled_data_points_, "./" + log_path_ + "/" + '_labelled_data_points.pt')
    torch.save(network_properties_, "./" + log_path_ + "/" + '_network_properties.pt')

    return problem_description_, collocation_points_, labelled_data_points_, network_properties_, log_path_, dir_path_


def main():
    
    # Initialize inputs
    problem_description, collocation_points, labelled_data_points, network_properties, log_path, dir_path = initialize_inputs()

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

    # Customize seed
    s = 123
    pl.seed_everything(s)

    # Initialize plot class
    plot = PlotClass()

    # Generate data
    dataModule = PINN_DataModule(problem_description, collocation_points, labelled_data_points, network_properties, log_path, dir_path)
    torch.save(dataModule, "./" + log_path + "/" + '_dataModule.pt')
    #with open("./" + log_path + "/" + '_dataModule.pkl', 'wb') as f:
    #    pickle.dump(dataModule, f)
    #dataModule.plotDistribution("./" + log_path)

    #dataModule.prepare_data()

    # Generate model
    model = Backbone(network_properties, problem_description)
    init_xavier(model)
    torch.save(model, "./" + log_path + "/" + '_model.pt')
    #with open("./" + log_path + "/" + '_model.pkl', 'wb') as f:
    #    pickle.dump(model, f)
    PINN_model = PINN_Model(model, network_properties, problem_description)

    # Define flags for callbacks, checkpoints, hardware, etc.
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="./", name=log_path)
    csvlogger   = pl_loggers.CSVLogger(save_dir="./", name=log_path)
    printingCallbacks = PrintingCallback()
    metricTracker     = MetricTracker()
    timer             = Timer()
    modelCheckpoints  = ModelCheckpoint(
        monitor='loss', 
        mode='min', 
        every_n_epochs=100, 
        save_on_train_epoch_end=True, 
        save_top_k=-1)
    callbacks = [printingCallbacks, 
                 metricTracker, 
                 modelCheckpoints, 
                 timer] #, EarlyStopping(monitor='loss_val', mode='min', verbose=True)]
    trainer = pl.Trainer(
        accelerator=network_properties["Device"].type,
        max_epochs=network_properties["Epochs"],
        default_root_dir="./",
        logger=[tensorboard, csvlogger],
        deterministic=True,
        callbacks=callbacks,
        inference_mode=False,
        check_val_every_n_epoch=100,
        enable_progress_bar=False,
        #profiler='simple'
        #log_every_n_steps=5
    )
    torch.save(trainer, "./" + log_path + "/" + '_trainer.pt')
    #with open("./" + log_path + "/" + '_trainer.pkl', 'wb') as f:
    #    pickle.dump(trainer, f)

    # Train the model
    print("##############   Fitting Model   ##############")
    timer.start_time("train")
    trainer.fit(PINN_model, datamodule=dataModule)
    elapsed_time_train = timer.time_elapsed("train")
    print("\nTraining Time: ", elapsed_time_train)
    ckpt_path = glob.glob("./" + log_path + "/version_0/checkpoints/" + "*.ckpt")[0]
    #plot.trainAndValidationErrors(metricTracker.collectionTrain, 'train', "./" + log_path)
    #plot.trainAndValidationErrors(metricTracker.collectionValidation, 'validation', "./" + log_path)

    #Testing the model
    #print("##############   Testing Model   ##############")
    #timer.start_time("test")
    #trainer.test(PINN_model, datamodule=dataModule, ckpt_path='best')
    #elapsed_time_test = timer.time_elapsed("test")
    #print("\nTesting Time: ", elapsed_time_test)

    # Make predictions
    print("##############   Evaluating Model   ##############")
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

