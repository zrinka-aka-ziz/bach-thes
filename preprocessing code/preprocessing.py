import phase1_sampling
import phase2_periodic_resample
import phase3_filter_wrong_sizes
import phase4_label_data

def execute_preprocessing(ext1, ext2):
	phase1_sampling.execute(ext1, ext2)
	phase2_periodic_resample.execute(ext1, ext2)
	phase3_filter_wrong_sizes.execute(ext1, ext2)
	phase4_label_data.execute(ext1, ext2)
	
execute_preprocessing('Train', 'train')
execute_preprocessing('Train', 'train2')
execute_preprocessing('Validation', 'validation')
execute_preprocessing('Test', 'test')

