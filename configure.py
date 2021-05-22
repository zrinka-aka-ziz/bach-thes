class Config:
	def __init__(self):
	#====data paths
	#	LOCAL
	#	self.path_short = "C:/Users/Jakov/Documents/Misc/FER - jakov/5. semestar/Projekt R/Materijali - segm. srca/Git-Projekt R/notebooks/UNet/"
	#	GOOGLE COLAB
		self.path_short = "/content/projektR/notebooks/UNet/"
		self.train_orig = self.path_short + "Train_images/train" #train dataset
		self.valid_orig = self.path_short + "Validation_images/validation" #validation dataset
		self.test_orig = self.path_short + "Test_images/test" #test dataset
		self.extension = "model0"
		self.expname = "Model 0 Only LAD Scaled size"
		self.checkpoints = self.path_short + self.extension + "/checkpoints/"
		self.optimizer = self.path_short + self.extension + "/optimizer/"
		self.test_results = self.path_short + self.extension + "/test/"
		
	#====data configure
		self.seed_value = 0
		self.imgsize = [848,848] #[1184,1184]
		self.masksize = [848,848] #[1184,1184]
		self.channels = 1
		self.threshold = 0.5
		self.train_batchsize = 4
		self.valid_batchsize = 4
		self.drop = 0.5
		self.LoadThread = 0
	#====training configure
		self.epochsize = 50
		self.lr = 0.00005 #initial learning rate
		self.gamma = 0.1
		self.lr_epoch_step = 30
		self.save_model_epoch = 2
		self.num_work = 0
		
		self.cuda_dev = 1
		self.cuda_dev_list = "0,1"
		
		
	#====preprocessing configure
	#====... placed in different file: preprocess_config