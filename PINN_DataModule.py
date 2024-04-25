
from ImportPackages import *
from miscelaneous import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pytorch_lightning.utilities import CombinedLoader
from pyDOE import lhs


class PINN_DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        problem_description = readInputData(config, "Problem description")
        network_properties = readInputData(config, "Network properties")
        collocation_points = readInputData(config, "Collocation points")
        labelled_data_points = readInputData(config, "Labelled data points")
        log_path = config["log_path"]
        dir_path = config["dir_path"]
        
        tags = ["Domain", "BoundaryCondition", "InitialCondition"]

        material_properties = problem_description["MaterialProperties"]
        lb = problem_description["SpaceAndTimeLowerBounds"]
        ub = problem_description["SpaceAndTimeUpperBounds"]
        ib_conditions = problem_description["InitialAndBoundaryConditions"]
        nonDimensional = problem_description["NonDimensional"]
        
        proportions = network_properties["DatasetPartitions"]
        batchSizeTrain = network_properties["BatchSizeTrain"]
        batchSizeValidation = network_properties["BatchSizeValidation"]
        batchSizeTest = network_properties["BatchSizeTest"]
        batchSizes = [batchSizeTrain, batchSizeValidation, batchSizeTest]
        self.BatchSizePredict = network_properties["BatchSizePredict"]
        device = network_properties["Device"]
        if device.type=='cpu':
            self.NumWorkers = 0 #os.cpu_count() - 1 #Use less number of cores than the total, 0 for no multiprocessing
            self.PinMemory = False
            self.PersistenWorkers = False
        else:
            self.NumWorkers = 1
            self.PinMemory = True
            self.PersistenWorkers = True
        
        self.ThermalDiffusivity = material_properties["ThermalDiffusivity"]
        self.BoundaryConditionValue = ib_conditions["BoundaryConditionValue"]
        self.ReferenceTemperature = material_properties["ReferenceTemperature"]
        self.NonDimensional = nonDimensional
        
        self.LB = torch.tensor(lb)
        self.UB = torch.tensor(ub)
        self.x_tensor = torch.linspace(lb[0], ub[0], 10000) 
        self.t_tensor = torch.linspace(lb[1], ub[1], 10000) 
        x_test = torch.linspace(lb[0], ub[0], 1000) 
        t_test = torch.linspace(lb[1], ub[1], 1000) 
        self.x_test = x_test
        self.t_test = t_test
        ms_x, ms_t = torch.meshgrid(x_test, t_test)
        x_test = ms_x.flatten().view(-1, 1)
        t_test = ms_t.flatten().view(-1, 1)
        self.X_test = torch.cat([x_test, t_test], axis=1)
        self.T_exact = exactSolution(self, x_test, t_test)
        
        N_dom = collocation_points['Domain']
        N_bc  = collocation_points['BoundaryCondition']
        N_ic  = collocation_points['InitialCondition']
        N_points = [N_dom, 2*N_bc, N_ic]
        
        useData = labelled_data_points["UseLabelledData"]
        if useData:
            N_points.append(labelled_data_points["NumberOfDataPoints"])
            tags.append("LabelledData")
        self.UseLabelledData = useData
        
        stages = ['train', 'validate', 'test', 'predict']
        listOfDicts = []
        for Ni in N_points:
            lengths = [int(np.round(p*Ni)) for p in proportions]
            sum_lengths = sum(lengths)
            if sum_lengths > Ni:
                idx = lengths.index(max(lengths))
                lengths[idx] -= sum_lengths - Ni
            listOfDicts.append(dict(zip(stages[0:3], lengths)))

        self.Proportions    = dict(zip(stages[0:3], proportions))
        self.BatchSizes     = dict(zip(stages[0:3], batchSizes))
        self.NumberOfPoints = dict(zip(tags, N_points))
        self.DatasetLengths = dict(zip(tags, listOfDicts))
        self.Tags   = tags
        self.Stages = stages

        self.BatchSizesTrainDict      = self.getRepresentativeBatchSizes('train')
        self.BatchSizesValidationDict = self.getRepresentativeBatchSizes('validate')
        self.BatchSizesTestDict       = self.getRepresentativeBatchSizes('test')

        self.LogPath = log_path
        self.DirPath = dir_path

    
    def prepare_data(self):
        if self.DirPath!=None:
            return 
        for stage in self.Stages:
            ds = self.buildDataset(stage)
            torch.save(ds, self.LogPath + '/' + stage + '.pt')
        

    def setup(self, stage: str):

        if stage == "fit":
            if self.DirPath==None:
                path_train = self.LogPath + '/train.pt'
                path_validation = self.LogPath + '/validate.pt'
            else:
                path_train = self.DirPath + '/train.pt'
                path_validation = self.DirPath + '/validate.pt'
            self.train = torch.load(path_train)
            self.validation = torch.load(path_validation)
        if stage == "validate":
            if self.DirPath==None:
                path_validation = self.LogPath + '/validate.pt'
            else:
                path_validation = self.DirPath + '/validate.pt'
            self.validation = torch.load(path_validation)
        
        if stage == "test":
            if self.DirPath==None:
                path_test = self.LogPath + '/test.pt'
            else:
                path_test = self.DirPath + '/test.pt'
            self.test = torch.load(path_test)

        if stage == "predict":
            if self.DirPath==None:
                path_predict = self.LogPath + '/predict.pt'
            else:
                path_predict = self.DirPath + '/predict.pt'
            self.predict = torch.load(path_predict)

        
    def train_dataloader(self):
    
        datasets = []
        for key in self.train:
            ds = DataLoader(dataset=self.train[key], batch_size=self.BatchSizesTrainDict[key],
                             shuffle=True, num_workers=self.NumWorkers, pin_memory=self.PinMemory, 
                             persistent_workers=self.PersistenWorkers)
            datasets.append(ds)
            
        loaders = dict(zip(self.Tags, datasets))
        combined_loader = CombinedLoader(loaders, mode='max_size_cycle')
        return combined_loader

    def val_dataloader(self):
        
        datasets = []
        for key in self.validation:
            ds = DataLoader(dataset=self.validation[key], batch_size=self.BatchSizesValidationDict[key], 
                            shuffle=False, num_workers=self.NumWorkers, pin_memory=self.PinMemory,
                            persistent_workers=self.PersistenWorkers)
            datasets.append(ds)
            
        loaders = dict(zip(self.Tags, datasets))
        combined_loader = CombinedLoader(loaders, mode='sequential')
        return combined_loader
    
    def test_dataloader(self):
        
        datasets = []
        for key in self.test:
            ds = DataLoader(dataset=self.test[key], batch_size=self.BatchSizesTestDict[key], 
                            shuffle=False, num_workers=self.NumWorkers, pin_memory=self.PinMemory, 
                            persistent_workers=self.PersistenWorkers)
            datasets.append(ds)
            
        loaders = dict(zip(self.Tags, datasets))
        combined_loader = CombinedLoader(loaders, mode='sequential')
        return combined_loader
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predict, batch_size=self.BatchSizePredict, 
                          shuffle=False, num_workers=self.NumWorkers, pin_memory=self.PinMemory, 
                          persistent_workers=self.PersistenWorkers)
        
    def buildDataset(self, stage):
        ds = []
        if stage=='predict':
            return CustomDataset(self.X_test, self.T_exact)
        for tag in self.Tags:
            if tag == "Domain":
                X = self.domain('LHS', stage, tag)
                y = torch.zeros((len(X), 1))
            if tag == "BoundaryCondition":
                X = self.boundaries(stage, tag)
                y = torch.zeros((len(X), 1))
            if tag == "InitialCondition":
                X = self.initialCondition(stage, tag)
                y = torch.zeros((len(X), 1))
            if tag == "LabelledData":
                X, y = self.labelledData(stage, tag)
                sigma = 0.05*torch.max(y)
                y = y + sigma*torch.randn_like(y)
            ds.append(CustomDataset(X, y))
        return dict(zip(self.Tags, ds))

    
    def domain(self, samplingType, stage, tag):
    
        N_dom = self.DatasetLengths[tag][stage]
        LB = self.LB
        UB = self.UB
        if samplingType=='LHS':
            # Generate collocation points within the domain using Latin Hypercube Sampling (LHS) strategy
            X_dom = (LB + (UB - LB)*lhs(2, N_dom)).to(torch.float32) 

        if samplingType=='Sobol':
            # Generate collocation points within the domain using Sobol sequence strategy
            raise ValueError('Sobol sequence not coded yet!')
        
        return X_dom
            
    def boundaries(self, stage, tag):
        
        # Generate random collocation points for the two BC from self.t_tensor
        N_bc = int(self.DatasetLengths[tag][stage]/2)
        LB = self.LB
        UB = self.UB
        t_tensor = self.t_tensor
        X = []
        x_limits = [LB[0], UB[0]]
        for x_limit in x_limits:
            idx_bc = np.random.choice(t_tensor.size()[0], N_bc, replace=False)
            t_bc = t_tensor[idx_bc].view(-1, 1)
            x_bc = x_limit*torch.ones((N_bc, 1))
            X.append(torch.cat([x_bc, t_bc], axis=1))

        X_bc = torch.cat([X[0], X[1]], axis=0)
        return X_bc
    
    def initialCondition(self, stage, tag):
    
        # Generate collocation points for the IC from self.x_tensor
        N_ic = self.DatasetLengths[tag][stage]
        x_tensor = self.x_tensor
        idx_ic = np.random.choice(x_tensor.size()[0], N_ic, replace=False)
        x_ic = self.x_tensor[idx_ic].view(-1, 1)
        t_ic = torch.zeros((N_ic, 1))
        X_ic = torch.cat([x_ic, t_ic], axis=1)
        return X_ic
        
    def labelledData(self, stage, tag):
        
        # Generate labelled data points within the domain using Latin Hypercube Sampling (LHS) strategy
        if self.UseLabelledData:
            N_data = self.DatasetLengths[tag][stage]
            LB = self.LB
            UB = self.UB
            X_data = (LB + (UB - LB)*lhs(2, N_data)).to(torch.float32)
            T_data = exactSolution(self, X_data[:, 0:1], X_data[:, 1:2])
            return X_data, T_data
        
        return None
        
    def getRepresentativeBatchSizes(self, stage):

        batch_size = self.BatchSizes[stage]
        if batch_size == -1:
            batch_size_by_source = [self.DatasetLengths[key][stage] for key in self.DatasetLengths]
        else:           
            proportions = []
            batch_size_by_source = []
            for key in self.NumberOfPoints:
                prop_i = self.NumberOfPoints[key]/sum(self.NumberOfPoints.values())
                size_i = batch_size*prop_i
                proportions.append(prop_i)
                batch_size_by_source.append(int(np.round(size_i)))
            sum_lengths = sum(batch_size_by_source)
            if sum_lengths > batch_size:
                idx = batch_size_by_source.index(max(batch_size_by_source))
                batch_size_by_source[idx] -= sum_lengths - batch_size
            if sum_lengths < batch_size:
                idx = batch_size_by_source.index(min(batch_size_by_source))
                batch_size_by_source[idx] += batch_size - sum_lengths

        return dict(zip(self.Tags, batch_size_by_source))
    


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx, :]
        target = self.y[idx, :]
        return features, target

