#==========================================================================
#=============================== Imports ==================================
#==========================================================================
import comet_ml
from comet_ml import Experiment

import re,os,time
import glob
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from torchsummary import summary
from tensorboardX import SummaryWriter

from configure import Config
config = Config()
from model import UNet
from DataLoader import ImageDataset
from loss import WBCELoss, TopKLoss, DiceLoss, GDiceLoss, DiceBCELoss, ELLoss


from torch.autograd import Variable
from simplejpeg import decode_jpeg
import albumentations as A
from natsort import natsorted,ns
import cv2

writer = SummaryWriter()

#==========================================================================
#======================= Experiment name setup ============================
#==========================================================================

experiment = Experiment(api_key="OsOrxAbb9TYm3nyP6QfY6F5QK", project_name="bach-thes", workspace="zrinka-aka-ziz")
experiment.log_parameters({"seed":config.seed_value, "input size":config.imgsize, "batch size train":config.train_batchsize,"batch size validation":config.valid_batchsize, "no of epochs":config.epochsize, "learning rate":config.lr, "learning rate decay":config.gamma, "learning rate decay step":config.lr_epoch_step, "dropout":config.drop})
experiment.set_name(config.expname)
#experiment.add_tags(["whole dataset","no dropout","with scheduler","full images","1,16,32,64,128,256"])


#==========================================================================
#========================== Define metrics ================================
#==========================================================================
#https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d

def get_acc(pred, orig):
	total = len(torch.flatten(orig))
	pred[pred <= config.threshold] = 0
	pred[pred > config.threshold] = 1
	correct = pred.eq(orig).sum()
	accuracy = correct.item()/total
	return accuracy

def get_prec(pred, orig):
	pred[pred <= config.threshold] = 0
	pred[pred > config.threshold] = 1
	cv = pred/orig
	tp = cv.eq(1).sum().item()
	fp = cv.eq(float('inf')).sum().item()
	tn = torch.isnan(cv).sum().item()
	fn = cv.eq(0).sum().item()
	prec = 0
	if tp + fp != 0:
		prec = tp/(tp+fp)
	return prec

def get_rec(pred, orig):
	pred[pred <= config.threshold] = 0
	pred[pred > config.threshold] = 1
	cv = pred/orig
	tp = cv.eq(1).sum().item()
	fp = cv.eq(float('inf')).sum().item()
	tn = torch.isnan(cv).sum().item()
	fn = cv.eq(0).sum().item()
	rec = 0
	if tp + fn != 0:
		rec = tp/(tp+fn)
	return rec

def get_iou(pred, orig):
	pred[pred <= config.threshold] = 0
	pred[pred > config.threshold] = 1
	cv = pred/orig
	tp = cv.eq(1).sum().item()
	fp = cv.eq(float('inf')).sum().item()
	tn = torch.isnan(cv).sum().item()
	fn = cv.eq(0).sum().item()
	iou_score = 0
	if tp + fp + fn != 0:
		iou_score = tp/(tp+fp+fn)
	return iou_score

#==========================================================================
#=============== Set everything for experiment repeatability ==============
#==========================================================================
np.random.seed(config.seed_value)
torch.manual_seed(config.seed_value) # cpu  vars
torch.cuda.manual_seed(config.seed_value)
torch.cuda.manual_seed_all(config.seed_value) # gpu vars
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False

#def _init_fn(worker_id):
#    np.random.seed(12 + worker_id)

#==========================================================================
#=========================== Initialization ===============================
#==========================================================================
#first dataset before updating
train_dataset = ImageDataset(config.train_orig) #Training Dataset
validation_dataset = ImageDataset(config.valid_orig) #Validation Dataset

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=config.train_batchsize, shuffle=True, num_workers= config.num_work, pin_memory=True, drop_last=True)
validation_loader = Data.DataLoader(dataset=validation_dataset, batch_size=config.valid_batchsize, shuffle=False, num_workers= config.num_work, pin_memory=True, drop_last=True)
## things needed for train2
imglist2 = natsorted(os.listdir(config.train2_orig+'/'), alg=ns.IGNORECASE)
resi = A.Resize(848,848,interpolation=2, always_apply=True, p=1)


iter = int((config.epochsize * len(train_dataset)) / config.train_batchsize)
iteri=0
iter_new=0 

model = UNet()
model.cuda()
summary(model,(config.channels,config.imgsize[0],config.imgsize[1]))

if config.extension=="model6":
    criterion = WBCELoss()
elif config.extension=="model7":
    criterion = TopKLoss()
elif config.extension=="model8":
    criterion = DiceLoss()
elif config.extension=="model9":
    criterion = GDiceLoss()
elif config.extension=="model10":
    criterion = DiceBCELoss()
elif config.extension=="model11":
    criterion = ELLoss()
else:
    criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=config.lr) #optimizer class
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_epoch_step, gamma=config.gamma)# this will decrease the learning rate by factor of 0.1 every 10 epochs

#==========================================================================
#========== Check if training starts from beggining or continues ==========
#==========================================================================
if os.path.exists(config.checkpoints) and len(os.listdir(config.checkpoints)):
    checkpoints = os.listdir(config.checkpoints)
    checkpoints.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
    model = torch.load(config.checkpoints+'/'+checkpoints[-1]) #changed to checkpoints
    iteri = int(re.findall(r'\d+',checkpoints[-1])[0]) # changed to checkpoints
    iter_new = iteri
    print("Resuming from iteration " + str(iteri))
elif not os.path.exists(config.checkpoints):
    os.makedirs(config.checkpoints)

if  os.path.exists(config.optimizer) and len(os.listdir(config.optimizer)):
    checkpoints = os.listdir(config.optimizer)
    checkpoints.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
    optimizer.load_state_dict(torch.load(config.optimizer+'/'+checkpoints[-1])) 
    print("Resuming Optimizer from iteration " + str(iteri))
elif not os.path.exists(config.optimizer):
    os.makedirs(config.optimizer)

#==========================================================================
#================================ TRAIN ===================================
#==========================================================================
beg=time.time() #time at the beginning of training
print("Training Started!")
with experiment.train():
  for epoch in range(config.epochsize):
    print("\nEPOCH " +str(epoch+1)+" of "+str(config.epochsize)+"\n")
    for i,datapoint_train in enumerate(train_loader):
      image_train = torch.as_tensor(datapoint_train['image']).type(torch.FloatTensor).cuda() #typecasting to FloatTensor as it is compatible with CUDA
      masks_train = torch.as_tensor(datapoint_train['mask']).type(torch.FloatTensor).cuda()

      optimizer.zero_grad()  #https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
      model.train()
      outputs_train = model(image_train)
      loss = criterion(outputs_train, masks_train)
      experiment.log_metric("loss", loss.item(), step=epoch*(len(train_dataset)/config.train_batchsize)+i)
      writer.add_scalar('Train loss', loss, iteri)
      loss.backward() #Backprop
      optimizer.step()    #Weight update
			
      iteri=iteri+1
			
      train_accuracy = get_acc(outputs_train, masks_train)
      writer.add_scalar('Train accuracy', train_accuracy, iteri)
      experiment.log_metric("accuracy", train_accuracy, step=epoch*(len(train_dataset)/config.train_batchsize)+i)
      
      train_prec = get_prec(outputs_train, masks_train)
      writer.add_scalar('Train precision', train_prec, iteri)
      experiment.log_metric("precision", train_prec, step=epoch*(len(train_dataset)/config.train_batchsize)+i)

      train_rec = get_rec(outputs_train, masks_train)
      writer.add_scalar('Train recall', train_rec, iteri)
      experiment.log_metric("recall", train_rec, step=epoch*(len(train_dataset)/config.train_batchsize)+i)

      train_iou = get_iou(outputs_train, masks_train)
      writer.add_scalar('Train IoU', train_iou, iteri)
      experiment.log_metric("IoU", train_iou, step=epoch*(len(train_dataset)/config.train_batchsize)+i)

  
		# Calculate Accuracy         
    validation_loss = 0
    validation_accuracy = 0
    validation_precision = 0
    validation_recall = 0
    validation_iou = 0
    model.eval()
    with torch.no_grad():
      with experiment.validate():
				# Iterate through validation dataset
        for j,datapoint_eval in enumerate(validation_loader): #for validation
          total_eval = 0
          correct_eval = 0
          image_eval = torch.as_tensor(datapoint_eval['image']).type(torch.FloatTensor).cuda()
          masks_eval = torch.as_tensor(datapoint_eval['mask']).type(torch.FloatTensor).cuda()

					# Forward pass only to get logits/output       
          outputs_eval = model(image_eval)
          validation_loss += criterion(outputs_eval, masks_eval).item()
          validation_accuracy += get_acc(outputs_eval, masks_eval)
          validation_precision += get_prec(outputs_eval, masks_eval)
          validation_recall += get_rec(outputs_eval, masks_eval)
          validation_iou += get_iou(outputs_eval, masks_eval)
        validation_accuracy = validation_accuracy / (j+1)
        validation_precision = validation_precision / (j+1)
        validation_recall = validation_recall / (j+1)
        validation_iou = validation_iou / (j+1)
        validation_loss = validation_loss / (j+1)

        writer.add_scalar('Validation loss', validation_loss, epoch)
        writer.add_scalar('Validation accuracy', validation_accuracy, epoch)
        writer.add_scalar('Validation precision', validation_precision, epoch)
        writer.add_scalar('Validation recall', validation_recall, epoch)
        writer.add_scalar('Validation IoU', validation_iou, epoch)

        experiment.log_metric("val_accuracy", validation_accuracy, step=epoch)
        experiment.log_metric("val_precision", validation_precision, step=epoch)
        experiment.log_metric("val_recall", validation_recall, step=epoch)
        experiment.log_metric("val_iou", validation_iou, step=epoch)
        experiment.log_metric("loss", validation_loss, step=epoch)
        
    time_since_beg = (time.time()-beg)/60
    experiment.log_other("time passed (min)", time_since_beg)
    # Print Loss
    print('Epoch: {}. Loss: {}. Validation Loss: {}. Time(mins) {}'.format(epoch, loss.item(), validation_loss,time_since_beg))
    # update training dataset
    print("Updating training dataset...")
    #delete masks from train2_masks
    #generate new masks for train2 using model
    #make imglist for train2
    #train2_dataset = ImageDataset(config.train2_orig) #Training Dataset
    #train2_loader = Data.DataLoader(dataset=train2_dataset, batch_size=config.train_batchsize, shuffle=True, num_workers= config.num_work, pin_memory=True, drop_last=True)
    #for image in imglist make mask
    
         #uzmi neku masku nije bitno koju, uzima se prva iz test
    with open(os.path.join(config.test_orig + '_masks/' + masklist[0]), 'rb') as f2:
        mask_or = decode_jpeg(f2.read())[:,:,1]
            
    for w in range(0,len(imglist),1):
        with open(os.path.join(config.train2_orig + '/' + imglist[w]), 'rb') as f1:
            image = decode_jpeg(f1.read())[:,:,1]
                #uzmi neku masku nije bitno koju, uzima se prva iz test
        
  
        resized = resi(image=image, mask=mask_or)
        inp = resized['image']
        #mask = resized['mask']

    #mask=mask_or
    #input_unet = image #- za full size
        inp = np.expand_dims(inp, axis=0)
        inp = np.expand_dims(inp, axis=0)
        #mask = np.expand_dims(mask, axis=0)
        inp.astype(float)
        inp = inp/ 255.0
        inp = torch.from_numpy(inp)
        inp = inp.type(torch.FloatTensor)
        inp = Variable(inp.cuda())

        #model_unet = torch.load(os.path.join(config.checkpoints+checkpoints))
        #model_unet.eval()
        #model_unet.cuda()
        outp = model(inp)
        oupt = outp.cpu().data.numpy()
        outp = outp * 255
        outp = outp.transpose((2, 3, 0, 1))
        outp = outp[:,:,:,0]
    
    #resized = res2(image=image,mask=out_unet)
    #out_unet  = resized['mask']

        outp[outp <= 127] = 0
        outp[outpt > 127] = 255

        outp.astype('uint8') #maska koju se treba dodati u training dataset prije iduce epohe
        cv2.imwrite(os.path.join(config.train2_orig + "_masks/" + imglist[w]), outp) #sta s ovim
    #upload masks to train2
    #copy train2 images to train  and train2 masks to train masks - merged version
#delete merged and copy train into it, then copy train2 into it
    
    #make new train loader that contains added masks
    train_dataset = ImageDataset(config.train_orig) #Training Dataset
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=config.train_batchsize, shuffle=True, num_workers= config.num_work, pin_memory=True, drop_last=True)
    scheduler.step()
    
    if epoch % config.save_model_epoch == 0:
      torch.save(model,config.checkpoints+'/model_ep_'+str(epoch)+'.pt')
      torch.save(optimizer.state_dict(),config.optimizer+'/model_ep_'+str(epoch)+'.pt')
#      print("model and optimizer saved at epoch : "+str(epoch))      
      
time_since_beg = (time.time()-beg)/60
experiment.log_parameters({"training time":time_since_beg})
print('Training time: {}'.format(time_since_beg))
writer.close()		

