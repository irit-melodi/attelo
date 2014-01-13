'''
Created on Jun 20, 2012

@author: stergos

contribs: phil

TODO: 
      - names are a little confusing
      x- reading and splitting of data should be separated in make_n_fold (first arg of make_n_fold should be a data table)

example usage for testing on some data: 
python  fileNfold.py ../../../data/attachement_relations_joint/all.csv


'''
import Orange


#from pprint import pprint
import random

# csv_file is the absolute path to the file that contains the data in Orange format
# the output is a dictionary whose keys are the files and values the test folds
#
#original_data = data.Table(csv_file)

def make_n_fold(original_data, folds = 5, meta_index = "FILE") :
	"""reads data file in csv/orange format, and provides a n-folds based
	on the meta index, in order not to mix instances from the same origin
	in train and test (eg, not to mix instances from the same file in train and test data
	in discourse experiment)

	returns the data in a table, and a dict of fold index by files
	"""
	
	file_index = original_data.domain.index(meta_index)
	fold_dict = dict()
	for i in range(len(original_data)) :
		file_key = original_data[i][file_index].value
		if not fold_dict.has_key(file_key) :
			fold_dict[file_key] = -1
	keys = fold_dict.keys()

	for current in xrange(((len(keys)) / folds) + 1) :
		random_values = random.sample(xrange(folds), folds)
		for i in xrange(folds) :
			position = (current * folds) + i
			if position < len(keys) :
				fold_dict[keys[position]] = random_values[i]


	return fold_dict


def makeFoldByFileIndex(data,fold_dict, meta_index = "FILE"):
	""" from folds based on meta_index and data, generate fold index for each instance
	fold_dict should be the result of make_n_fold
	the index can then be used by data.select to make train/test data without any mixing
	of instances from the same meta_index

	NB: fold index for instances allows for the use of any normal method of orange as if it were classical cross-validation
	"""
	index = []
	file_index = data.domain.index(meta_index)
	for one in data:
		file_key = one[file_index].value
		index.append(fold_dict[file_key])
	return index



def process_by_file_folds(data,selection,f_train,f_test,f_eval,folds = 5):
	"""
	just for demo/example purposes
	could be a template of processing a corpus based on folds-on-files
	
	data: a data table
	selection: an index of folds for each instance, eg can be computed with
	               selection= orange.MakeRandomIndicesCV(data, folds=5)
	f_train/ f_test: what to do on training/testing
	f_eval: evaluate one fold
	folds = number of folds. must be the same as the nb in selection or else ouch. (should/could be recomputed from selection)
	"""
	evals = []
	for test_fold in range(folds):
		train_data = data.select_ref(selection, test_fold, negate=1)
		test_data = data.select_ref(selection, test_fold)
		model = f_train(train_data)
		results = f_test(test_data,model)
		evals.append(f_eval(results))
	# or evals+results
	return evals

# testing with a simple 5-fold cross-validation
if __name__=="__main__":
	import sys

	#random.seed()#"just an illusion")
	
	bayes = Orange.classification.bayes.NaiveLearner(adjust_threshold=True)
	data = Orange.data.Table(sys.argv[1])
	fold_struct = make_n_fold(data,folds = 5)
	#print fold_struct
	#sys.exit(0)

	selection = makeFoldByFileIndex(data,fold_struct)
	
	print ">>>> 5-fold with file awareness"
	results = Orange.evaluation.testing.test_with_indices([bayes],data,selection)
	print "accuracy =", Orange.evaluation.scoring.CA(results)[0]
	cm = Orange.evaluation.scoring.confusion_matrices(results)[0]
	print "F1 for class True", Orange.evaluation.scoring.F1(cm)
	
	# comparaison avec nfold sans tri par fichier
	print ">>>> 5-fold without file awareness"
	results = Orange.evaluation.testing.cross_validation([bayes],data,folds=5)
	print "accuracy =", Orange.evaluation.scoring.CA(results)[0]
	cm = Orange.evaluation.scoring.confusion_matrices(results)[0]
	print "F1 for class True", Orange.evaluation.scoring.F1(cm)
	
	# test distrib
	nb = bayes(data)
	predicted_class, distribution = nb(data[0],result_type=Orange.classification.Classifier.GetBoth)
	
	

