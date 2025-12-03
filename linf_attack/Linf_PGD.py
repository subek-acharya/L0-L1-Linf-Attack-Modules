import torch
import utils

def ProjectionOperation(xAdv, xClean, epsilonMax):
    xAdv = torch.min(xAdv, xClean + epsilonMax)
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

def PGDNativePytorch(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, loss_type="ce"):
    model.eval()
    
    # Select loss function based on type
    if loss_type == "ce":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_type == "dlr":
        loss_fn = utils.dlr_loss
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Use 'ce' or 'dlr'.")
    
    # Generate variables for storing adversarial examples
    numSamples = len(dataLoader.dataset)
    xShape = utils.GetOutputShape(dataLoader)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    
    # Process each batch
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        
        for attackStep in range(numSteps):
            xAdvCurrent.requires_grad = True
            output = model(xAdvCurrent)
            model.zero_grad()
            
            # Calculate loss based on selected type
            if loss_type == "ce":
                cost = loss_fn(output, yCurrent)
            else:  # dlr
                cost = loss_fn(output, yCurrent)
                cost = cost.mean()  # DLR returns vector, need to take mean
            
            cost.backward()
            advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)
            advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        
        # Save adversarial images
        for j in range(batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex += 1
    
    advLoader = utils.TensorToDataLoader(
        xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None
    )
    return advLoader