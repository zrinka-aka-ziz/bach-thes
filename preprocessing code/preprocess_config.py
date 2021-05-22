class PreprocessConfig:
	def __init__(self):
		#====preprocessing configure
		self.patch_size = 64
		self.no_of_patches = 19
		self.resample_rate = 4
		
		self.black_white_treshold = 10
		
		
		
		#MAKE SURE THESE PATHS IN "configure.py" and "preprocess_config.py" match!
		
		#Google Colab
		self.path_short ="/content/projektR/notebooks/UNet/"
		
		#Local
		#self.path_short ="C:/Users/Jakov/Documents/Misc/FER - jakov/5. semestar/Projekt R/Materijali - segm. srca/Git-Projekt R/notebooks/UNet/"