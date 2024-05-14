
from ImportPackages import *
from miscelaneous import *
from ActivationAndInitializationFunctions import activation

class Backbone(nn.Module):
    
    def __init__(self, network_properties, problem_description):
        super().__init__()

        self.InputDimensions      = network_properties["InputDimensions"]
        self.OutputDimensions     = network_properties["OutputDimensions"]
        self.NumberOfHiddenLayers = network_properties["HiddenLayers"]
        self.NumberOfNeurons      = network_properties["NumberOfNeurons"]
        self.NumberOfEpochs       = network_properties["Epochs"]
        self.ActFunctionString    = network_properties["Activation"]
        self.Device               = network_properties["Device"]
        
        self.NonDimensional         = problem_description["NonDimensional"]
        self.LeftBoundaryCondition  = problem_description["InitialAndBoundaryConditions"]["LeftBoundaryCondition"]
        self.RightBoundaryCondition = problem_description["InitialAndBoundaryConditions"]["RightBoundaryCondition"]
        self.HardImposedDirichletBC = problem_description["HardImposedDirichletBC"]
        lb = problem_description["SpaceAndTimeLowerBounds"]
        ub = problem_description["SpaceAndTimeUpperBounds"]
        self.LB = torch.tensor(lb, device=self.Device)
        self.UB = torch.tensor(ub, device=self.Device)

        self.InputLayer = nn.Linear(self.InputDimensions, self.NumberOfNeurons)

        self.HiddenLayers = nn.ModuleList(
            [nn.Linear(self.NumberOfNeurons, self.NumberOfNeurons) for _ in range(self.NumberOfHiddenLayers - 1)])
        self.OutputLayer = nn.Linear(self.NumberOfNeurons, self.OutputDimensions)

        self.ActivationFunction = activation(self.ActFunctionString)
        
    def forward(self, x):
        x = 2*(x - self.LB)/(self.UB - self.LB) - 1

        out = self.ActivationFunction(self.InputLayer(x))
        for k, l in enumerate(self.HiddenLayers):
            out = self.ActivationFunction(l(out))

        out = self.OutputLayer(out)
        if self.HardImposedDirichletBC:
            if self.LeftBoundaryCondition["Type"] != "Dirichlet":
                raise ValueError("Hard imposition of Neumann or Robin conditions not coded yet!")
            elif self.RightBoundaryCondition["Type"] != "Dirichlet":
                raise ValueError("Hard imposition of Neumann or Robin conditions not coded yet!")

            x0   = x[:, 0:1]
            lb0 = 2*(self.LB[0] - self.LB[0])/(self.UB[0] - self.LB[0]) - 1
            ub0 = 2*(self.UB[0] - self.LB[0])/(self.UB[0] - self.LB[0]) - 1
            Hcal = self.heaviside(x0, lb0, ub0)
            one  = torch.ones_like(x0)
            bc_l = self.LeftBoundaryCondition["Value"]*one
            bc_r = self.RightBoundaryCondition["Value"]*one
            xlen = ub0 - lb0
            dx   = x0 - lb0*one
            bc   = bc_l + (bc_r - bc_l)/xlen*dx
            one  = torch.ones_like(out)
            out  = bc*(one-Hcal) + out*Hcal 

        return out
    
    def heaviside(self, x, lb, ub):
        eps = 1e-6
        x1 = (lb + eps)*torch.ones_like(x)
        x2 = (ub - eps)*torch.ones_like(x)
        one = torch.ones_like(x)
        d1 =  x - x1
        d2 =  x2 - x
        dist = torch.minimum(d1, d2)
        Hcal = 0.5*(one + dist/eps + 1.0/torch.pi*torch.sin(dist*torch.pi/eps) )
        xout = torch.where(torch.less(dist, -eps), torch.zeros_like(x), torch.where(torch.greater(dist, eps), torch.ones_like(x), Hcal))
        return xout


class PINN_Model(pl.LightningModule):

    def __init__(self, backbone, problem_description, network_properties):
        super().__init__()

        self.backbone = backbone
        
        self.OptimizerString     = network_properties["Optimizer"]
        self.CriterionString     = network_properties["Criterion"]
        self.LearningRate        = network_properties["LearningRate"]
        self.WeightDecay         = network_properties["WeightDecay"]
        self.Device              = network_properties["Device"]
        self.BatchSizeTrain      = network_properties["BatchSizeTrain"]
        self.BatchSizeValidation = network_properties["BatchSizeValidation"]
        self.BatchSizeTest       = network_properties["BatchSizeTest"]
        
        material_properties = problem_description["MaterialProperties"]
        physical_domain     = problem_description["PhysicalDomain"]
        time_domain         = problem_description["TimeDomain"]
        lb                  = problem_description["SpaceAndTimeLowerBounds"]
        ub                  = problem_description["SpaceAndTimeUpperBounds"]
        ib_conditions       = problem_description["InitialAndBoundaryConditions"]
        heat_source         = problem_description["HeatSource"]
        nonDimensional      = problem_description["NonDimensional"]
        
        leftBC  = ib_conditions["LeftBoundaryCondition"]
        rightBC = ib_conditions["RightBoundaryCondition"]
        self.Density = float(material_properties["Density"])
        self.ThermalConductivity = float(material_properties["ThermalConductivity"] )
        self.SpecificHeatCapacity = float(material_properties["SpecificHeatCapacity"])
        self.ThermalDiffusivity = float(material_properties["ThermalDiffusivity"])
        self.ReferenceTemperature = float(material_properties["ReferenceTemperature"])
        self.InitialTemperature = ib_conditions["InitialCondition"]
        self.LeftBoundaryCondition = leftBC
        self.RightBoundaryCondition = rightBC
        self.DomainLowerBounds = lb
        self.DomainUpperBounds = ub
        self.HeatSource = heat_source
        self.NonDimensional = nonDimensional

        self.NbrOfTrainBatches = []
        self.NbrOfValidationBatches = [[] for _ in range(4)]
        self.NbrOfTestBatches = [[] for _ in range(4)]
        self.train_keys   = ['loss', 'loss_train_pde', 'loss_train_bc', 'loss_train_ic', 'loss_train_data']
        self.train_losses = [0 for _ in range(len(self.train_keys))]
        self.val_keys     = ['loss_val', 'loss_val_pde', 'loss_val_bc', 'loss_val_ic', 'loss_val_data']
        self.val_losses   = [0 for _ in range(len(self.val_keys))]
        self.test_keys    = ['loss_test', 'loss_test_pde', 'loss_test_bc', 'loss_test_ic', 'loss_test_data']
        self.test_losses  = [0 for _ in range(len(self.test_keys))]


    def training_step(self, train_batch, batch_idx):
        
        loss_list = []
        #print('Train batch_idx: ', batch_idx)
        for key in train_batch:
            X, y = train_batch[key]
            if key=='Domain':
                if X==None:
                    y, y_hat = 0, 0
                    continue
                X.requires_grad = True
                T = self.backbone(X) 
                y_hat = self.governingEquationsResidue(X, T)
            if key=='BoundaryCondition':
                if X==None:
                    y, y_hat = 0, 0
                    continue
                T = self.backbone(X) 
                y_hat = self.boundaryConditionsResidue(X, T)
            if key=='InitialCondition':
                if X==None:
                    y, y_hat = 0, 0
                    continue
                T = self.backbone(X) 
                y_hat = self.initialConditionResidue(X, T)
            if key=='LabelledData':
                if X==None:
                    y, y_hat = 0, 0
                    continue
                y_hat = self.backbone(X) 
                
            loss_by_key = self.configure_criterion(y_hat, y) 
            loss_list.append(loss_by_key)
        
        loss = sum(loss_list)
        loss_list.insert(0, loss)
        #print('loss_list: ', loss_list)
        if 'LabelledData' not in train_batch:
            loss_list.append(0)
        
        train_keys = self.train_keys
        log_dict = dict(zip(train_keys, loss_list))
        #self.log_dict(log_dict, prog_bar=True)
        self.NbrOfTrainBatches.append(batch_idx)
        return log_dict 

    def on_train_batch_end(self, out, train_batch, batch_idx):
        len_out = len(out)
        for i, key in enumerate(out):
            self.train_losses[i] += out[key] #.append(out[key])
        
    def on_train_epoch_end(self):
        avg_loss_keys = self.train_keys
        if self.train_losses[-1]==0:
            avg_loss_keys.pop()
        
        avg_loss_values = []
        batches = self.NbrOfTrainBatches
        #print('self.train_losses: ', self.train_losses)
        #print('avg_loss_keys: ', avg_loss_keys)
        for i in range(len(avg_loss_keys)):
            avg_loss_by_key = self.train_losses[i]/len(batches)
            avg_loss_values.append(avg_loss_by_key)
        
        log_dict = dict(zip(avg_loss_keys, avg_loss_values))
        self.log_dict(log_dict, sync_dist=True) #, prog_bar=True)
        self.NbrOfTrainBatches = []
        self.train_losses = [0 for _ in range(len(avg_loss_keys))]
        aux = "".join(f"{key}: {value:.5e}, " for key, value in log_dict.items())
        print(f"Epoch: {self.current_epoch}, " + aux)

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        
        X, y = val_batch
        torch.set_grad_enabled(True)
        if dataloader_idx==0:
            X.requires_grad = True
            T = self.backbone(X)
            y_hat = self.governingEquationsResidue(X, T)
            #y_hat = torch.tensor([0.0, 0.0])
            #y = torch.tensor([0.0, 0.0])
        if dataloader_idx==1:
            T = self.backbone(X)
            y_hat = self.boundaryConditionsResidue(X, T)
        if dataloader_idx==2:
            T = self.backbone(X)
            y_hat = self.initialConditionResidue(X, T)
        if dataloader_idx==3:
            y_hat = self.backbone(X)
            
        loss = self.configure_criterion(y_hat, y) 
        self.NbrOfValidationBatches[dataloader_idx].append(batch_idx)
        return loss 
    
    def on_validation_batch_end(self, out, val_batch, batch_idx, dataloader_idx=0):
        for i in range(len(self.val_losses)-1):
            if dataloader_idx==i:
                self.val_losses[i+1] += out

    def on_validation_epoch_end(self):
        avg_loss_keys = self.val_keys
        if self.val_losses[-1]==0:
            avg_loss_keys.pop()
        
        avg_loss = 0
        avg_loss_values = []
        batches = self.NbrOfValidationBatches
        #print(self.val_losses)
        for i in range(len(avg_loss_keys)-1):
            avg_loss_by_key = self.val_losses[i+1]/len(batches[i])
            avg_loss_values.append(avg_loss_by_key)
            avg_loss += avg_loss_by_key
        
        avg_loss_values.insert(0, avg_loss)
        log_dict = dict(zip(avg_loss_keys, avg_loss_values))
        self.log_dict(log_dict, sync_dist=True)
        self.NbrOfValidationBatches = [[] for _ in range(4)]
        self.val_losses = [0 for _ in range(len(avg_loss_keys))]
        aux = "".join(f"{key}: {value:.5e}, " for key, value in log_dict.items())
        print(f"Epoch: {self.current_epoch}, " + aux)


    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):

        X, y = test_batch
        torch.set_grad_enabled(True)
        if dataloader_idx==0:
            X.requires_grad = True
            T = self.backbone(X)
            y_hat = self.governingEquationsResidue(X, T)
        if dataloader_idx==1:
            T = self.backbone(X)
            y_hat = self.boundaryConditionsResidue(X, T)
        if dataloader_idx==2:
            T = self.backbone(X)
            y_hat = self.initialConditionResidue(X, T)
        if dataloader_idx==3:
            y_hat = self.backbone(X)

        loss = self.configure_criterion(y_hat, y)
        self.NbrOfTestBatches[dataloader_idx].append(batch_idx)
        return loss     
        
    def on_test_batch_end(self, out, test_batch, batch_idx, dataloader_idx=0):
        for i in range(len(self.test_losses)-1):
            if dataloader_idx==i:
                self.test_losses[i+1] += out
            
    def on_test_epoch_end(self):
        avg_loss_keys = self.test_keys
        if self.test_losses[-1]==0:
            avg_loss_keys.pop()
        
        avg_loss = 0
        avg_loss_values = []
        batches = self.NbrOfTestBatches
        #print(self.test_losses)
        for i in range(len(avg_loss_keys)-1):
            avg_loss_by_key = self.test_losses[i+1]/len(batches[i])
            avg_loss_values.append(avg_loss_by_key)
            avg_loss += avg_loss_by_key
        
        avg_loss_values.insert(0, avg_loss)
        log_dict = dict(zip(avg_loss_keys, avg_loss_values))
        self.log_dict(log_dict)
        self.NbrOfTestBatches = [[] for _ in range(4)]
        self.test_losses = [0 for _ in range(len(avg_loss_keys))]
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        T_hat = self.backbone(X)
        return T_hat

    def configure_optimizers(self):
        if self.OptimizerString=="ADAM":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.LearningRate,
                                         weight_decay=self.WeightDecay)
        else:
            raise ValueError('Not coded yet!')
        
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
        
    def configure_criterion(self, y_hat, y):
        if self.CriterionString=="MSE":
            criterion = nn.MSELoss(reduction='mean')
        else:
            raise ValueError('Not coded yet!')

        return criterion(y_hat, y)
    
    def governingEquationsResidue(self, X, T):
        
        rho = self.Density
        k = self.ThermalConductivity
        c = self.SpecificHeatCapacity
        rhoc = rho*c
    
        # Compute the derivatives of the output w.r.t. the inputs using AD mechanism:
        diff_T = torch.autograd.grad(T, X,
                                    create_graph=True,
                                    grad_outputs=torch.ones_like(T))[0]
        
        Tx, Ty, Tt = diff_T[:, 0:1], diff_T[:, 1:2], diff_T[:, 2:3]

        div_kTx = torch.autograd.grad(k*Tx, X,
                                    create_graph=True,
                                    grad_outputs=torch.ones_like(T))[0][:, 0:1]
        
        div_kTy = torch.autograd.grad(k*Ty, X,
                                    create_graph=True,
                                    grad_outputs=torch.ones_like(T))[0][:, 1:2]

        q = self.heatSource(X)
        residue = rhoc*Tt - (div_kTx + div_kTy) - q

        return residue 
    
    def boundaryConditionsResidue(self, X, T):
        
        x = X[:, 0:1]
        k = self.Conductivity
        BC_left = self.LeftBoundaryCondition
        BC_right = self.RightBoundaryCondition
        typeOfBC = [BC_left["Type"], BC_right["Type"]]
        
        x_left  = x==self.DomainLowerBounds[0]
        x_right = x==self.DomainUpperBounds[0]
        x_boundary = [x_left, x_right]
        
        bc_values = torch.zeros_like(x)
        residue = torch.zeros_like(x)
        bc_values[x_left]  = float(BC_left["Value"])
        bc_values[x_right] = float(BC_right["Value"])
        
        for i, bc in enumerate(typeOfBC):
            idx = x_boundary[i].flatten()
            if bc=='Dirichlet':
                residue[idx, :] = T[idx, :] - bc_values[idx, :]
                continue
            
            T_x = torch.autograd.grad(T, X,
                                        create_graph=True,
                                        grad_outputs=torch.ones_like(T))[0][:, 0:1]
            
            if bc=='Neumann':
                residue[idx, :] = bc_values[idx, :] + k*T_x[idx, :]
                
            if bc=='Robin':
                if BC_left.Type==bc:
                    T_infty = float(BC_left["EnvironmentTemperature"])
                if BC_right.Type==bc:
                    T_infty = float(BC_right["EnvironmentTemperature"])
                residue[idx, :] = bc_values[idx, :]*(T[idx, :] - T_infty) + k*T_x[idx, :]
            
        return residue

    def initialConditionResidue(self, X, T):
        
        x = X[:, 0:1]
        ic_value = self.InitialTemperature(x)
        residue = T - ic_value
        return residue

    def heatSource(self, X):

        x = X[:, 0:1]
        y = x[:, 1:2]
        r0 = X[:, 3:4]
        v = self.HeatSource["Velocity"]
        P = self.HeatSource["TotalPower"]
        timestep = -0. + x*v
        q =  P/(torch.pi*r0**2)*torch.exp(-((x - timestep)**2 + y**2)/r0**2)
        return q