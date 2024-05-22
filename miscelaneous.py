
import torch
import json
import os


def readInputData(config, key):

    currentPath = config["parent_path"]
    with open(currentPath + 'inputData.json') as json_file:
        inputData = json.load(json_file)
    
    for keys2tune in config.keys():
        if keys2tune in inputData:
            inputData[keys2tune]["Value"] = config[keys2tune]

    if key=='Problem description':
        rho = inputData["Material properties"]["Density"]["Value"]
        k = inputData["Material properties"]["ThermalConductivity"]["Value"]
        cp = inputData["Material properties"]["SpecificHeatCapacity"]["Value"]
        alpha = k/(rho*cp)
        refTemp = inputData["Material properties"]["ReferenceTemperature"]["Value"]
        xi = inputData["Physical domain"]["LeftCoordinate"]["Value"]
        xf = inputData["Physical domain"]["RightCoordinate"]["Value"]
        ti = inputData["Time domain"]["InitialTime"]["Value"]
        tf = inputData["Time domain"]["FinalTime"]["Value"]
        non_dimensional = bool(inputData["Non dimensional analysis"]["Value"])
        hardImposedDirichletBC = bool(inputData["Hard imposed Dirichlet boundary conditions"]["Value"])
        rho_ref, k_ref, cp_ref, alpha_ref, T_ref, kx, kt = 1, 1, 1, 1, 1, 1, 1
        if non_dimensional:
            rho_ref, k_ref, cp_ref, alpha_ref, T_ref = rho, k, cp, alpha, refTemp
            kx, kt = 1/xf, k_ref/(rho_ref*cp_ref*(xf-xi)**2)
        
        # Define data structures 
        material_properites_ = {
            "Dict_Name": "Material properties",
            "Density": rho/rho_ref, 
            "ThermalConductivity": k/k_ref, 
            "SpecificHeatCapacity": cp/cp_ref, 
            "ThermalDiffusivity": alpha/alpha_ref,
            "ReferenceTemperature": refTemp/T_ref # K
        }
        physical_domain_ = {
            "Dict_Name": "Physical domain",
            "LeftCoordinate": xi*kx, # meters
            "RightCoordinate": xf*kx # meters
        }
        time_domain_ = {
            "Dict_Name": "Time domain",
            "InitialTime": ti*kt, # seconds
            "FinalTime": tf*kt # seconds
        }
        L = physical_domain_["RightCoordinate"] - physical_domain_["LeftCoordinate"]
        bc_temp_value = inputData["Boundary conditions"]["LeftBoundaryCondition"]["Value"]/T_ref 
        T0 = material_properites_["ReferenceTemperature"]
        ib_conditions_ = {
            "Dict_Name": "Initial and boundary conditions",
            "InitialCondition": lambda x: T0*torch.sin(torch.pi*x/L) + bc_temp_value, # K
            "BoundaryConditionValue": bc_temp_value,
            "LeftBoundaryCondition": {
                'Type': inputData["Boundary conditions"]["LeftBoundaryCondition"]["Type"],
                'Value': inputData["Boundary conditions"]["LeftBoundaryCondition"]["Value"]/T_ref 
                },
            "RightBoundaryCondition": {
                'Type': inputData["Boundary conditions"]["RightBoundaryCondition"]["Type"],
                'Value': inputData["Boundary conditions"]["RightBoundaryCondition"]["Value"]/T_ref 
                }
        }

        outputDict = {
            "Dict_Name": key,
            "MaterialProperties": material_properites_,
            "PhysicalDomain": physical_domain_,
            "TimeDomain": time_domain_,
            "SpaceAndTimeLowerBounds": [float(physical_domain_["LeftCoordinate"]), float(time_domain_["InitialTime"])],
            "SpaceAndTimeUpperBounds": [float(physical_domain_["RightCoordinate"]), float(time_domain_["FinalTime"])],
            "InitialAndBoundaryConditions": ib_conditions_,
            "NonDimensional": non_dimensional,
            "HardImposedDirichletBC": hardImposedDirichletBC,
        }
        return outputDict

    inputDict = inputData[key]
    if key=='Collocation points':
        outputDict = {
            "Dict_Name": key,
            "Domain": inputDict["Domain"]["Value"],
            "BoundaryCondition": inputDict["BoundaryCondition"]["Value"],
            "InitialCondition": inputDict["InitialCondition"]["Value"]
        }

    elif key=='Labelled data points':
        outputDict = {
            "Dict_Name": key,
            "UseLabelledData": bool(inputDict["UseLabelledData"]),
            "NumberOfDataPoints": inputDict["NumberOfDataPoints"]["Value"]
        }

    elif key=='Network properties':
        device_type = inputDict["Device"]["Value"]
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
        
        outputDict = {
            "Dict_Name": key,
            "InputDimensions": inputDict["InputDimensions"]["Value"],
            "OutputDimensions": inputDict["OutputDimensions"]["Value"],
            "HiddenLayers": inputDict["HiddenLayers"]["Value"],
            "NumberOfNeurons": inputDict["NumberOfNeurons"]["Value"],
            "DatasetPartitions": inputDict["DatasetPartitions"]["Value"],
            "WeightDecay": inputDict["WeightDecay"]["Value"],
            "Epochs": inputDict["Epochs"]["Value"],
            "LearningRate": inputDict["LearningRate"]["Value"],
            "Activation": inputDict["Activation"]["Value"],
            "Optimizer": inputDict["Optimizer"]["Value"],
            "Criterion": inputDict["Criterion"]["Value"],
            "Device": device,
            "BatchSizeTrain": inputDict["BatchSizeTrain"]["Value"],
            "BatchSizeValidation": inputDict["BatchSizeValidation"]["Value"],
            "BatchSizeTest": inputDict["BatchSizeTest"]["Value"],
            "BatchSizePredict": inputDict["BatchSizePredict"]["Value"]
        }

    else:
        print("Key: ", key)
        raise ValueError('Invalid key!')

    return outputDict
    

def exactSolution(obj, x, t):
    '''
    The following is the exact solution for the 1D heat equation with 
    Dirichlet boundary conditions of value T_bc at each end of the bar,
    and half sinusiodal initial condition of amplitude A.
    '''

    T_bc  = obj.BoundaryConditionValue
    A     = obj.ReferenceTemperature
    alpha = obj.ThermalDiffusivity
    lb    = obj.LB
    ub    = obj.UB
    L = ub[0] - lb[0]
    
    T = A*torch.sin(torch.pi*x/L)*torch.exp(-torch.pi**2*alpha*t/L**2) + T_bc
    return T.view(-1, 1)


