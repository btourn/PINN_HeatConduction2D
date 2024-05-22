
from ImportPackages import *
from miscelaneous import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pytorch_lightning.utilities import CombinedLoader
from pyDOE import lhs
import sobol_seq


class PINN_DataModule(pl.LightningDataModule):

    def __init__(self, problem_description, collocation_points, labelled_data_points, network_properties, log_path, dir_path):
        super().__init__()

        tags = ["Domain", "BoundaryCondition", "InitialCondition"]

        material_properties = problem_description["MaterialProperties"]
        physical_domain     = problem_description["PhysicalDomain"]
        time_domain         = problem_description["TimeDomain"]
        lb                  = problem_description["VariableLowerBounds"]
        ub                  = problem_description["VariableUpperBounds"]
        ib_conditions       = problem_description["InitialAndBoundaryConditions"]
        heat_source         = problem_description["HeatSource"]
        nonDimensional      = problem_description["NonDimensional"]
        
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
        
        self.PhysicalDomain = physical_domain
        self.TimeDomain = time_domain
        self.ThermalDiffusivity = material_properties["ThermalDiffusivity"]
        self.ReferenceTemperature = material_properties["ReferenceTemperature"]
        self.NonDimensional = nonDimensional
        self.HeatSource = heat_source
        
        self.LB = torch.tensor(lb)
        self.UB = torch.tensor(ub)
        self.x_tensor = torch.linspace(lb[0], ub[0], 10000) 
        self.y_tensor = torch.linspace(lb[1], ub[1], 10000) 
        self.t_tensor = torch.linspace(lb[2], ub[2], 10000) 
        self.r_tensor = torch.linspace(lb[3], ub[3], 10000) 
        
        #self.X_test = torch.cat([x_test, y_test], axis=1)
        #self.T_exact = exactSolution(self, x_test, t_test)
        
        N_dom = collocation_points['Domain']
        N_bc  = collocation_points['BoundaryCondition']
        N_ic  = collocation_points['InitialCondition']
        prop  = collocation_points['ProportionOfEntriesWithinDisk']
        rd    = collocation_points['RadiusOfDisk']
        N_points = [N_dom, 4*N_bc, N_ic]
        
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

        self.ProportionOfEntriesWithinDisk = prop
        self.RadiusOfDisk = rd

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
            return CustomDataset(0, 0) #CustomDataset(self.X_test, self.T_exact)
        for tag in self.Tags:
            if tag == "Domain":
                X = self.domain('Sobol', stage, tag)
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
            XY_dom = (LB + (UB - LB)*lhs(len(LB), N_dom)).to(torch.float32) 

        if samplingType=='Sobol':
            # Generate collocation points within the domain using Sobol sequence strategy
            dataset = self.generateData(tag, N_dom)
            XY_dom = torch.from_numpy(dataset).type(torch.FloatTensor)
        
        return XY_dom
            
    def boundaries(self, stage, tag):
        
        # Generate random collocation points for the two BC from self.t_tensor
        N_bc = int(self.DatasetLengths[tag][stage]/4)
        LB = self.LB
        UB = self.UB
        x_tensor = self.x_tensor
        y_tensor = self.y_tensor
        t_tensor = self.t_tensor
        r_tensor = self.r_tensor
        X, Y = [], []
        x_limits = [LB[0], UB[0]]
        y_limits = [LB[1], UB[1]]
        for x_limit in x_limits:
            idx_bc = np.random.choice(t_tensor.size()[0], N_bc, replace=False)
            x_bc = x_limit*torch.ones((N_bc, 1))
            y_bc = y_tensor[idx_bc].view(-1, 1)
            t_bc = t_tensor[idx_bc].view(-1, 1)
            r_bc = r_tensor[idx_bc].view(-1, 1)
            X.append(torch.cat([x_bc, y_bc, t_bc, r_bc], axis=1))

        for y_limit in y_limits:
            idx_bc = np.random.choice(t_tensor.size()[0], N_bc, replace=False)
            x_bc = x_tensor[idx_bc].view(-1, 1)
            y_bc = y_limit*torch.ones((N_bc, 1))
            t_bc = t_tensor[idx_bc].view(-1, 1)
            r_bc = r_tensor[idx_bc].view(-1, 1)
            Y.append(torch.cat([x_bc, y_bc, t_bc, r_bc], axis=1))

        X_bc = torch.cat([X[0], X[1]], axis=0)
        Y_bc = torch.cat([Y[0], Y[1]], axis=0)
        XY_bc = torch.cat([X_bc, Y_bc], axis=0)
        return XY_bc
    
    def initialCondition(self, stage, tag):
    
        # Generate collocation points for the IC from self.x_tensor
        N_ic = self.DatasetLengths[tag][stage]
        dataset = self.generateData(tag, N_ic)
        XY_ic = torch.from_numpy(dataset).type(torch.FloatTensor)
        return XY_ic


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


    def generateData(self, tag, N):

        skip = 0
        prop = self.ProportionOfEntriesWithinDisk
        ring_thick = self.RadiusOfDisk
        s_ring_rad = 0.0
        n = int(N*prop) #number of total center-bias points

        xi = self.PhysicalDomain["LeftCoordinate"]
        xf = self.PhysicalDomain["RightCoordinate"]
        yi = self.PhysicalDomain["BottomCoordinate"]
        yf = self.PhysicalDomain["TopCoordinate"]

        ti = self.TimeDomain["InitialTime"]
        tf = self.TimeDomain["FinalTime"]

        x0 = self.HeatSource["InitialXPosition"] #Initial position of laser over x-axis
        y0 = self.HeatSource["InitialYPosition"] #Initial position of laser over y-axis
        t0 = self.HeatSource["InitialTime"] #Time where transient analysis begins
        vs = self.HeatSource["Velocity"] #Velocity of heat source
        r0_i = self.HeatSource["LowerCharacteristicRadius"] 
        r0_f = self.HeatSource["UpperCharacteristicRadius"]

        data_x = np.full((N, 1), np.nan)
        data_y = np.full((N, 1), np.nan)

        if tag=='Domain':
            data_t = (ti + (tf - ti))*lhs(1, N)
        elif tag=='InitialCondition':
            data_t = np.full((N, 1), t0)
        else:
            raise ValueError('Invalid tag!')

        if r0_i==r0_f:
            data_r = r0_i*np.ones_like(data_x)
        else:
            data_r = (r0_i + (r0_f - r0_i))*lhs(1, N)
        data_x[:N-n-1, 0] = np.random.triangular(xi, x0 + data_t[:N-n-1, 0]*vs, xf) #majority of coll points sampled around laser
        
        while True: #Loop the ensure that the samples from the Laplace distribution are within the limits (yi, yf)
            aux = np.random.laplace(y0, 0.1*(yf+abs(yi))/2, N-n-1)
            aux_min = np.min(aux)
            aux_max = np.max(aux)
            if (aux_min>=yi) and (aux_max<=yf):
                break
        data_y[:N-n-1, 0] = aux
        for j in range(N-n-1, N): #Center-bias: some points directly in laser center
            seed = j + skip
            rnd, _ = sobol_seq.i4_sobol(3, seed)
            radi =  s_ring_rad + rnd[0]*ring_thick
            theta = rnd[1]*np.pi/2 
            phi = rnd[2]*2*np.pi
            coord_x = x0 + (data_t[j, :] - t0)*vs + radi*np.cos(phi)*np.sin(theta)
            #coord_x = x0 + radi*np.cos(phi)*np.sin(theta)
            if (coord_x<xi):
                coord_x = -coord_x
            elif (coord_x>xf):
                coord_x = xf - coord_x
            data_x[j, :] = coord_x
            data_y[j, :] = y0 + radi*np.sin(phi)*np.sin(theta)

        dataset = np.concatenate((data_x, data_y, data_t, data_r), axis=1)
        return dataset

        
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

