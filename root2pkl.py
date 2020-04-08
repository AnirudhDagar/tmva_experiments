from root_numpy import root2array
import pickle

# Load data into numpy arrays
input_file = "/Users/gollum/Downloads/sample_images_32x32.root"
sig1 = root2array(input_file, 'sig_tree;1')
sig2 = root2array(input_file, 'sig_tree;2')
bkg1 = root2array(input_file, 'bkg_tree;1')
bkg2 = root2array(input_file, 'bkg_tree;2')

# Open pkl file to dump the tree arrays
with open("root_test_data.pkl", 'w+') as file
	pickle.dump(sig1, file)
	pickle.dump(sig2, file)
	pickle.dump(bkg1, file)
	pickle.dump(bkg2, file)
