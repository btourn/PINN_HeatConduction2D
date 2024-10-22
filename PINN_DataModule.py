
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

        material_properties   = problem_description["MaterialProperties"]
        physical_domain       = problem_description["PhysicalDomain"]
        time_domain           = problem_description["TimeDomain"]
        lb                    = problem_description["VariableLowerBounds"]
        ub                    = problem_description["VariableUpperBounds"]
        ib_conditions         = problem_description["InitialAndBoundaryConditions"]
        heat_source           = problem_description["HeatSource"]
        nonDimensional        = problem_description["NonDimensional"]
        nonDimensionalFactors = problem_description["NonDimensionalFactors"]
        
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
        self.NonDimensionalFactors = nonDimensionalFactors
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
        
        stages_tags = ['train', 'validate', 'test', 'predict']
        listOfDicts = []
        stages = []
        for Ni in N_points:
            lengths = [int(np.round(p*Ni)) for p in proportions]
            sum_lengths = sum(lengths)
            if sum_lengths > Ni:
                idx = lengths.index(max(lengths))
                lengths[idx] -= sum_lengths - Ni
            listOfDicts.append(dict(zip(stages_tags[0:3], lengths)))

        stages = [True]*len(stages_tags)
        for i, p in enumerate(proportions):
            if p==0:
                stages[i] = False

        self.ProportionOfEntriesWithinDisk = prop
        self.RadiusOfDisk = rd

        self.Proportions    = dict(zip(stages_tags[0:3], proportions))
        self.BatchSizes     = dict(zip(stages_tags[0:3], batchSizes))
        self.Stages         = dict(zip(stages_tags, stages))
        self.NumberOfPoints = dict(zip(tags, N_points))
        self.DatasetLengths = dict(zip(tags, listOfDicts))
        self.Tags           = tags
        self.PredictKey     = None
        self.AllKeysPredict = None

        self.BatchSizesTrainDict      = self.getRepresentativeBatchSizes('train')
        self.BatchSizesValidationDict = self.getRepresentativeBatchSizes('validate')
        self.BatchSizesTestDict       = self.getRepresentativeBatchSizes('test')

        self.LogPath = log_path
        self.DirPath = dir_path

        self.save_hyperparameters()

    
    def prepare_data(self):
        if self.DirPath!=None:
            return 
        for stage in self.Stages.keys():
            if self.Stages[stage]>0:
                fname = self.LogPath + '/' + stage + '.pt'
                if os.path.isfile(fname):
                    continue
                else:
                    ds = self.buildDataset(stage)
                    if stage=='predict':
                        self.AllKeysPredict = list(ds.keys())
                    torch.save(ds, self.LogPath + '/' + stage + '.pt')
                    for key in ds:
                        torch.save(ds[key].X, self.LogPath + '/' + stage + '_' +  key + '.pt')
        

    def setup(self, stage: str):

        if stage == "fit":
            if self.DirPath==None:
                path_train = self.LogPath + '/train.pt'
                try: 
                    path_validation = self.DirPath + '/validate.pt'
                except:
                    path_validation = None
            else:
                path_train = self.DirPath + '/train.pt'
                try: 
                    path_validation = self.DirPath + '/validate.pt'
                except:
                    path_validation = None
            self.train = torch.load(path_train)
            try: 
                self.validation = torch.load(path_validation)
            except:
                self.validation = None

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
                path_predict = self.LogPath + '/predict_' + self.PredictKey + '.pt'
            else:
                try:
                    path_predict = self.DirPath + '/predict.pt'
                except:
                    path_predict = None
            try:
                self.predict = torch.load(path_predict)
            except:
                self.predict = None

        
    def train_dataloader(self):
    
        datasets = []
        for key in self.train:
            ds = DataLoader(dataset=self.train[key], batch_size=self.BatchSizesTrainDict[key],
                             shuffle=False, num_workers=self.NumWorkers, pin_memory=self.PinMemory, 
                             persistent_workers=self.PersistenWorkers)
            datasets.append(ds)
            
        loaders = dict(zip(self.Tags, datasets))
        combined_loader = CombinedLoader(loaders, mode='max_size_cycle')
        return combined_loader

    '''def val_dataloader(self):
        
        datasets = []
        for key in self.validation:
            ds = DataLoader(dataset=self.validation[key], batch_size=self.BatchSizesValidationDict[key], 
                            shuffle=False, num_workers=self.NumWorkers, pin_memory=self.PinMemory,
                            persistent_workers=self.PersistenWorkers)
            datasets.append(ds)
            
        loaders = dict(zip(self.Tags, datasets))
        combined_loader = CombinedLoader(loaders, mode='sequential')
        return combined_loader'''

    '''def test_dataloader(self):
        
        datasets = []
        for key in self.test:
            ds = DataLoader(dataset=self.test[key], batch_size=self.BatchSizesTestDict[key], 
                            shuffle=False, num_workers=self.NumWorkers, pin_memory=self.PinMemory, 
                            persistent_workers=self.PersistenWorkers)
            datasets.append(ds)
            
        loaders = dict(zip(self.Tags, datasets))
        combined_loader = CombinedLoader(loaders, mode='sequential')
        return combined_loader'''
    
    def predict_dataloader(self):

        ds = self.predict
        bs = self.BatchSizePredict
        if bs==-1:
            bs = len(ds)
        return DataLoader(dataset=ds, batch_size=bs, 
                          shuffle=False, num_workers=self.NumWorkers, pin_memory=self.PinMemory, 
                          persistent_workers=self.PersistenWorkers)
    
    def buildDataset(self, stage):
        ds = []
        if stage=='predict':
            ts_ = [5, 25, 45]
            r0_ = [2, 5, 8]
            return self.generatePredictDatasets(ts_, r0_) 
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
            
    def boundaries(self, stage=None, tag=None, N_bc=None):
        
        # Generate random collocation points for the two BC from self.t_tensor
        if N_bc==None:
            aux = self.DatasetLengths[tag][stage]
            N_bc = int(self.DatasetLengths[tag][stage]/4)
        else:
            aux = N_bc
            N_bc = int(N_bc/4)
        
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

        i = 0
        for i, y_limit in enumerate(y_limits):
            i += 1
            if i==2:
                if 4*N_bc!=aux:
                    N_bc += aux - 4*N_bc
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
        N0 = int(N_ic*.8) #80% of points completely inside the domain
        N1 = N_ic - N0 #Remaining points exactly at the boundaries
        dataset0 = self.generateData(tag, N0)
        XY_ic0 = torch.from_numpy(dataset0).type(torch.FloatTensor)
        XY_ic1 = self.boundaries(N_bc=N1)
        XY_ic1[:, 2:3] = torch.zeros(len(XY_ic1), 1) #Impose t=0
        XY_ic = torch.cat([XY_ic0, XY_ic1], axis=0)
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

        x0_s = self.HeatSource["InitialXPosition"] #Initial position of laser over x-axis
        y0_s = self.HeatSource["InitialYPosition"] #Initial position of laser over y-axis
        ti_s = self.HeatSource["InitialTime"] #Time where transient analysis begins
        tf_s = self.HeatSource["FinalTime"]
        vs = self.HeatSource["Velocity"] #Velocity of heat source
        r0_i = self.HeatSource["LowerCharacteristicRadius"] 
        r0_f = self.HeatSource["UpperCharacteristicRadius"]

        data_x = np.full((N, 1), np.nan)
        data_y = np.full((N, 1), np.nan)

        if tag=='Domain':
            
            xf_s = x0_s + vs*(tf_s-ti_s)
            t_before = ti_s - ti
            t_after  = tf - tf_s
            case0_t = (t_before==0) and (t_after!=0)
            case1_t = (t_before!=0) and (t_after==0)
            case2_t = (t_before!=0) and (t_after!=0)
            case3_t = (t_before==0) and (t_after==0)
            if case0_t:
                t_aux1 = ti + (tf - ti)*lhs(1, N-n) #tf_s + (tf - tf_s)*lhs(1, N-n)
                t_aux_s = ti + (tf_s - ti)*lhs(1, n)
                #x_aux_s = np.random.triangular(x0_s, x0_s + (t_aux_s-ti_s)*vs, xf_s)
                x_aux_s = x0_s + (xf_s - x0_s)*lhs(1, len(t_aux_s))
                data_t = np.concatenate((t_aux1, t_aux_s), axis=0)
            elif case1_t:
                t_aux1 = ti + (tf - ti)*lhs(1, N-n) #ti + (ti_s - ti)*lhs(1, N-n)
                t_aux_s = ti_s + (tf - ti_s)*lhs(1, n)
                #x_aux_s = np.random.triangular(x0_s, x0_s + (t_aux_s-ti_s)*vs, xf_s)
                x_aux_s = x0_s + (xf_s - x0_s)*lhs(1, len(t_aux_s))
                data_t = np.concatenate((t_aux1, t_aux_s), axis=0)
            elif case2_t:
                N1 = int((tf_s - ti_s)/(tf - ti)*(N-n))
                N2 = N - N1 - n
                #t_aux1 = ti + (ti_s - ti)*lhs(1, N1)
                #t_aux2 = tf_s + (tf - tf_s)*lhs(1, N2)
                t_aux1 = ti + (tf - ti)*lhs(1, N-n) #
                t_aux_s = ti_s + (tf_s - ti_s)*lhs(1, n)
                #x_aux_s = np.random.triangular(x0_s, x0_s + (t_aux_s-ti_s)*vs, xf_s)
                x_aux_s = x0_s + (xf_s - x0_s)*lhs(1, len(t_aux_s))
                #data_t = np.concatenate((t_aux1, t_aux2, t_aux_s), axis=0)
                data_t = np.concatenate((t_aux1, t_aux_s), axis=0)
            elif case3_t:
                data_t = ti + (tf - ti)*lhs(1, N)

            x_before = x0_s - xi
            x_after  = xf - xf_s
            case0_x = (x_before==0) and (x_after!=0)
            case1_x = (x_before!=0) and (x_after==0)
            case2_x = (x_before!=0) and (x_after!=0)
            case3_x = (x_before==0) and (x_after==0)
            if case0_x:
                #x_aux1 = np.random.triangular(xi, xi + t_aux1*vs, xf)
                x_aux1 = xi + (xf - xi)*lhs(1, N-n)
                data_x = np.concatenate((x_aux1, x_aux_s), axis=0)
            elif case1_x:
                #x_aux1 = np.random.triangular(xi, xi + t_aux1*vs, xf)
                x_aux1 = xi + (xf - xi)*lhs(1, N-n)
                data_x = np.concatenate((x_aux1, x_aux_s), axis=0)
            elif case2_x:
                #x_aux1 = np.random.triangular(xi, xi + t_aux1*vs, xf) #(xi + (xf - xi))*lhs(1, N-n)
                x_aux1 = xi + (xf - xi)*lhs(1, N-n)
                data_x = np.concatenate((x_aux1, x_aux_s), axis=0)
            elif case3_x:
                data_x[:N-n] = xi + (xf - xi)*lhs(1, N-n) #data_x = np.random.triangular(x0_s, x0_s + data_t*vs, xf_s)
            #data_t = (ti + (tf - ti))*lhs(1, N)

            while True: #Loop the ensure that the samples from the Laplace distribution are within the limits (yi, yf)
                aux = np.random.laplace(y0_s, 0.1*(yf+abs(yi))/2, N-n-1)
                aux_min = np.min(aux)
                aux_max = np.max(aux)
                if (aux_min>=yi) and (aux_max<=yf):
                    break
            data_y[:N-n-1, 0] = aux

        elif tag=='InitialCondition':
            data_t = np.full((N, 1), ti)
            data_x = np.full((N, 1), np.nan)
            data_x[:N-n] = xi + (xf - xi)*lhs(1, N-n)
            data_y[:N-n] = yi + (yf - yi)*lhs(1, N-n)
            vs = 0
            ring_thick *= 5
        else:
            raise ValueError('Invalid tag!')

        if r0_i==r0_f:
            data_r = r0_i*np.ones_like(data_x)
        else:
            data_r = r0_i + (r0_f - r0_i)*lhs(1, N)
        
        
        for j in range(N-n-1, N): #Center-bias: some points directly in laser center
            seed = j + skip
            rnd, _ = sobol_seq.i4_sobol(3, seed)
            radi =  s_ring_rad + rnd[0]*ring_thick
            theta = rnd[1]*np.pi/2 
            phi = rnd[2]*2*np.pi
            coord_x = x0_s + (data_t[j, :] - ti_s)*vs + radi*np.cos(phi)*np.sin(theta)
            #coord_x = x0 + radi*np.cos(phi)*np.sin(theta)
            if (coord_x<xi):
                coord_x = -coord_x
            elif (coord_x>xf):
                coord_x = xf - coord_x
            data_x[j, :] = coord_x
            data_y[j, :] = y0_s + radi*np.sin(phi)*np.sin(theta)

        dataset = np.concatenate((data_x, data_y, data_t, data_r), axis=1)
        np.random.shuffle(dataset)
        return dataset
    

    def generatePredictDatasets(self, ts_, r0_):

        nonDimensionalFactors = self.NonDimensionalFactors
        heatSource = self.HeatSource
        vs = heatSource["Velocity"]
        kx = nonDimensionalFactors["Space"]
        kt = nonDimensionalFactors["Time"]
        xy_data = self.PhysicalDomain

        eps_x = 1e-3*xy_data["RightCoordinate"]
        eps_y = 1e-3*xy_data["TopCoordinate"]
        ys = 0
        y0 = -np.flipud(np.geomspace(eps_y, xy_data["TopCoordinate"], 50))
        y1 = np.geomspace(eps_y, xy_data["TopCoordinate"], 50)
        y  = np.concatenate((y0, np.array([ys]), y1), axis=0)
        x  = np.linspace(xy_data["LeftCoordinate"], xy_data["RightCoordinate"], 250)
        
        ds = []
        keys = []
        ts = [kt*x for x in ts_]
        xs = [kt*x*vs for x in ts_]
        r0 = [kx*x for x in r0_]
        name = "ds_"
        for i, tsi in enumerate(ts):
            xt = torch.from_numpy(x).type(torch.FloatTensor)
            yt = torch.from_numpy(y).type(torch.FloatTensor)
            ms_x, ms_y = torch.meshgrid(xt, yt)
            x_pred = ms_x.flatten().view(-1, 1)
            y_pred = ms_y.flatten().view(-1, 1)
            t_pred = torch.ones_like(x_pred)*tsi
            for j, r0j in enumerate(r0):
                r_pred = torch.ones_like(x_pred)*r0j
                XY_pred = torch.cat([x_pred, y_pred, t_pred, r_pred], axis=1)
                T_hat = torch.full(r_pred.size(), torch.nan)
                s = str(r0_[j]) 
                aux = s.replace('.', '') + "_" + str(ts_[i]) + "s"
                keys.append(name + aux)
                ds.append(CustomDataset(XY_pred, T_hat))
                
        return dict(zip(keys, ds))


        
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

