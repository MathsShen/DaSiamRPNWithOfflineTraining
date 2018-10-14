
class ModelConfig(object):

	def __init__(self):
		self.image_format = 'jpeg'

		self.batch_size = 5
		self.max_seq_len = 15

		self.image_size = [224, 224]
		self.num_image_channels = 3

		self.num_clstm_kernels = 256 #384 for two layers of ConvLSTMs
		self.clstm_kernel_size = [3, 3]
		self.num_convlstm_layers = 2
		# If < 1.0, the dropout keep probability applied to ConvLSTM variables.
		self.clstm_dropout_keep_prob = 0.7

		self.pretrained_model_file = None

		self.training_data_tfrecord_path = \
			'/home/lishen/Experiments/CLSTMT/dataset/training_set/TFRecord/training_data.tf_record.soft_gt'

		# Approximate number of values per input shard. Used to ensure sufficient
		# mixing between shards in training.
		self.values_per_input_shard = 2300
		# Minimum number of shards to keep in the input queue.
		self.input_queue_capacity_factor = 2
		# Number of threads for prefetching SequenceExample protos.
		self.num_input_reader_threads = 1

		# Number of threads for image preprocessing. Should be a multiple of 2.
		self.num_preprocess_threads = 1

		self.num_seqs = 115 # total number of domains(training tracking sequences)


class TrainingConfig(object):

	def __init__(self):
		"""Set the default training hyper-parameters."""

		# Optimizer for training the model
		self.optimizer = "SGD"
		self.max_epoches = 100
		self.learning_rate = 0.001


class FinetuningConfig(object):
	def __init__(self):
		self.learning_rate = 0.01
		self.use_domain_specific_finetuned_model = True


class TestingConfig(object):

	def __init__(self):
		self.root_dir = '/home/lishen/Experiments/CLSTMT'
		self.code_root_dir = '/home/code/lishen/dataset'
		self.peep_ratio = 3.5


class VerificationModelConfig(object):
	def __init__(self):
		self.pretrained_model_file = "./weights/vgg16_verif.npy"
		self.num_boxes_per_batch = None

		
