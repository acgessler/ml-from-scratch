
var models = models || {};

// Restricted Boltzmann Machine.
//
// Learns a joint probability distribution of `num_visible_units` binary-valued inputs
// using `num_hidden_units` binary units as memory. When applied to image data sets,
// visible units (with one exception explained below) refer to image pixels being
// "on" or "off" (to handle grayscale).
//
// A RBN is a special case of a Markov Random Field (MRF) where the visible and hidden
// inputs are fully connected amongst each other but not amongst themselves. This
// structure allows efficient use of (Block) Gibbs Sampling to obtain samples of
// the visible or the hidden distributions. Unlike general Feed-Forward Neural Networks,
// RBMs are generative: given any sample from the input distribution, the RBN can
// provide the most likely reconstruction.
//
// This RBN also supports class labels (which are just another visible inputs) for
// classification. Each class label is a discrete-valued number that are mapped to
// a vector of bionary, one-hot inputs. For classification of a given input, the class
// label is omitted and the expected value of the visible distribution is calculated.
// This yields a probability for each class label, the highest of which becomes the
// prediction.
//
// num_visible_units: Number of visible units.
// num_hidden_units: Number of hidden units.
// num_label_classes: Number of label casses. The class label is represented as
//   this many additional visible units which are treated as an one-hot vector.
//
// References:
//  [Hinton 2010] "A Practical Guide to Training Restricted Boltzmann Machines"
//    Geoffrey Hinton, 2010 (https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
models.RBN = function(num_visible_units, num_hidden_units, num_label_classes) {
	this.num_hidden_units = num_hidden_units;
	this.num_visible_units = num_visible_units;
	this.num_label_classes = num_label_classes;
	this.num_visible_units_and_labels = num_visible_units + this.num_label_classes;

	// Weights and bias are floating-point numbers.
	// Initial values follow the recommendations in [Hinton 2010]
	this.weights = this.create2DArray_(Float32Array,
		this.num_visible_units_and_labels,
		this.num_hidden_units,
		function() {
			return util.SampleNormal(0.0, 0.01);
		});
	this.visible_bias = this.create1DArray_(Float32Array,
		this.num_visible_units_and_labels,
		function() {
			// This is later updated in initVisibleBias_().
			return 0.0;
		});
	this.hidden_bias = this.create1DArray_(Float32Array,
		this.num_hidden_units,
		function() {
			return 0.0;
		});

	// Field below store state during training. To avoid repeated allocations, we
	// pre-create them and hold them in the RBN instance (The disadvantage is that
	// it makes the data flow during training a bit less obvious).

	// Current activation for visible and hidden units. Activations are binary (0
	// or 1). However, to reduce sampling noise we occasionally use the expected
	// value of the activation, which is a floating-point value between 0 and 1.
	this.visible_activations = new Float32Array(this.num_visible_units_and_labels);
	this.hidden_activations = new Float32Array(this.num_hidden_units);
	// Class activations are at the end of the `visible_activations`. Most of the
	// time, they require no special handling. When they do, we access them using
	// this convenience sub-array.
	this.class_activations = this.visible_activations.subarray(this.num_visible_units);
	this.weights_gradient = this.create2DArray_(Float32Array,
		this.num_visible_units_and_labels,
		this.num_hidden_units);
	this.visible_bias_gradient = this.create1DArray_(Float32Array,
		this.num_visible_units_and_labels);
	this.hidden_bias_gradient = this.create1DArray_(Float32Array,
		this.num_hidden_units);

	// Used by evalReconstructionError().
	this.old_visible_activations = new Float32Array(this.num_visible_units_and_labels);
	this.started_training = false;
};


// Train on the given labeled training set.
// num_batches: Number of training batches.
// batch_size: Number of (randomly chosen) examples per batch, defaults to 1.
// gibbs_sampling_steps: Markov Chain length for Gibbs sampling (CD-k), defaults to 1.
// learning_rate: Multiplier for weight updates, defaults to 0.001.
//
models.RBN.prototype.train = function(labeled_examples,  num_batches, batch_size,
	gibbs_sampling_steps,
	learning_rate) {

	// Config defaults.
	gibbs_sampling_steps = gibbs_sampling_steps || 1;
	learning_rate = learning_rate || 0.001;
	batch_size = batch_size || 1;

	// The first time this runs, initialize visible biases from training data.
	if (!this.started_training) {
		this.initVisibleBias_(labeled_examples);
		this.started_training = true;
	}

	for (var batch = 0; batch < num_batches; ++batch) {
		this.clearGradients_();
		// Run `batch_size` training examples.
		// Each example adds to the positive and the negative gradients.
		for (var n = 0; n < batch_size; ++n) {
			// Get a random training example from `labeled_images`.
			var training_example = util.RandomElement(labeled_examples);
			this.updateGradientsFromExample_(training_example, gibbs_sampling_steps);	
		}
		// Tune weights using learning_rate * (positive_gradient - negative_gradient) / batch_size.
		// This is effectively gradient-descent back-propagation.
		this.applyWeightUpdates_(learning_rate / batch_size);
		// TODO: update learning_rate? (Adagrad)
	}
	this.assertFiniteness_();
};


// Compute evaluation error for the classification task on the given `labeled_examples`.
// num_samples: Number of (randomly chosen) examples to evaluate. Defaults to
//   |labeled_examples|. If -1 is passed, no sampling is performed and all input
//   examples are classified.
//
models.RBN.prototype.evalClassificationError = function(labeled_examples, num_samples) {
	var all = num_samples == -1;
	num_samples = all ? labeled_examples.length : (num_samples || labeled_examples.length);
	var count_correct = 0;
	for (var i = 0; i < num_samples; ++i) {
		var labeled_example = all ? labeled_examples[i] : util.RandomElement(labeled_examples);
		var true_label = labeled_example.getClassLabel();
		var predicted_label = this.classifyExample(labeled_example);
		if (true_label == predicted_label) {
			++count_correct;
		}
	}
	return 1.0 - count_correct / num_samples;
};


// Estimate cross entropy loss between the original and the reconstruction.
// This does not utilize example labels.
// num_samples: Number of (randomly chosen) examples to evaluate. Defaults to
//   |labeled_examples|. If -1 is passed, no sampling is performed and all input
//   examples are classified.
//
models.RBN.prototype.evalReconstructionError = function(examples, num_samples) {
	var all = false;
	num_samples = all ? examples.length : (num_samples || examples.length);
	var mean_reconstruction_xent = 0.0;
	for (var i = 0; i < num_samples; ++i) {
		var example = all ? examples[i] : util.RandomElement(examples);
		var reconstruction_xent = this.reconstructionErrorForExample(example);
		// Numerically stable calculation of mean cross-entropy reconstruction loss.
		mean_reconstruction_xent += (reconstruction_xent - mean_reconstruction_xent) / (i + 1);
	};
	return mean_reconstruction_xent;
};


// Calculate the Cross-Entropy loss between the original and the reconstruction.
// This does not utilize example labels.
models.RBN.prototype.reconstructionErrorForExample = function(example) {
	this.setVisibleActivationFromExample_(example);
	this.old_visible_activations.set(this.visible_activations);
	// Perform one step back and forth of conditional sampling.
	this.sampleHiddenFromVisibleUnits_();
	// Keep expected value (probability) since it plays well with Cross-Entropy loss.
	this.sampleVisibleFromHiddenUnits_(true);

	return util.CrossEntropy(
		this.old_visible_activations.subarray(0, this.num_visible_units),
		this.visible_activations.subarray(0, this.num_visible_units));
};


// Classify the given example using the trained model.
// Returns the index of the predicted class label.
//
models.RBN.prototype.classifyExample = function(example) {
	this.setVisibleActivationFromExample_(example);
	// Perform one step back and forth of conditional sampling.
	this.sampleHiddenFromVisibleUnits_();
	// Keep expected value in class labels. We want to take the class with the highest
	// probability, not a sampled category.
	this.sampleVisibleFromHiddenUnits_(/* use_expected_value */ true);

	var max_class_activation = 0.0;
	var predicted_class_index = 0;
	for (var class_index = 0; class_index < this.class_activations; ++class_index) {
		if (this.class_activations[class_index] > max_class_activation) {
			max_class_activation = this.class_activations[class_index];
			predicted_class_index = class_index;
		}
	}
	return predicted_class_index;
};


// Reconstruct visible unit state for the given example.
// labeled_example: Labeled input example. However, an unlabeled example is also accepted.
// Returns an UInt8Array containing binary state for each visible unit (not including
// units pertaining to class labels). This represents an unbiased sample of the
// distribution learnt by the RBN, conditioned on the input example.
//
models.RBN.prototype.reconstructVisibleUnitsForExample = function(labeled_or_unlabeled_example) {
	if (labeled_or_unlabeled_example.GetClassLabel) {
		this.setVisibleActivationFromLabeledExample_(labeled_or_unlabeled_example);
	}
	else {
		this.setVisibleActivationFromExample_(labeled_or_unlabeled_example);
	}
	this.sampleHiddenFromVisibleUnits_();
	this.sampleVisibleFromHiddenUnits_();
	return new Uint8Array(this.visible_activations.slice(0, this.num_visible_units));
};


// Get the filter represented by a single hidden unit h_i.
// The return value is a Float32Array() containing `num_visible_units` numbers
// between 0 and 1.
models.RBN.prototype.getFilterForHiddenUnit = function(hidden_unit_index) {
	var filter = new Float32Array(this.num_visible_units);
	for (var vi = 0; vi < this.num_visible_units; ++vi) {
		filter[vi] = util.Sigmoid(this.weights[vi][hidden_unit_index] +
			this.hidden_bias[hidden_unit_index]);
	}
	return filter;
};


// Initialize visible biases based on a sample of the training examples.
// This is called before the first training step to speed up initial training by
// initializing visible biases as to reproduce the activation probabilities of each
// visible unit across the training set.
//
models.RBN.prototype.initVisibleBias_ = function(labeled_examples) {
	var NUM_SAMPLES = 100;
	// Initialize the visible bias to be the log-odds of the corresponding input being set
	// in the training data (-> [Hinton 2010]): Since initial weights w_ij are drawn from a
	// normal distribution centered at 0, the initial reconstruction of the visible unit v_i
	// does not need to utilize the hidden units to reproduce the activation probability p_i
	// of the training vectors since
	//      E(v_i) = E(sigmoid(logit(p_i) + sum_j(h_j * N(0, _)))) = p_i
	this.visible_bias.fill(0.0);
	for (var i = 0; i < NUM_SAMPLES; ++i) {
		var example = util.RandomElement(labeled_examples);
		this.setVisibleActivationFromLabeledExample_(example);
		// Mis-use visible_bias as an accumulator.
		for (var j = 0; j < this.num_visible_units_and_labels; ++j) {
			this.visible_bias[j] += this.visible_activations[j];
		}
	}
	for (var k = 0; k < this.num_visible_units_and_labels; ++k) {
		var activation_probability = this.visible_bias[k] / NUM_SAMPLES;
		this.visible_bias[k] = activation_probability > 0 ? util.Logit(activation_probability) : 0;
	}
};


// Set visible activations from the given (unlabeled) `example`.
//
models.RBN.prototype.setVisibleActivationFromExample_ = function(example) {
	// This does not populate class labels, which are at the end of the array.
	this.visible_activations.fill(0.0);
	this.visible_activations.set(example.getBinaryFeatures());
};


// Set visible activations from the given `labeled_example`.
// This does the same as setVisibleActivationFromExample_ but also sets the label class activation.
//
models.RBN.prototype.setVisibleActivationFromLabeledExample_ = function(labeled_example) {
	this.setVisibleActivationFromExample_(labeled_example);
	// Class labels are categorical, i.e., exactly one activation is set.
	var true_label = labeled_example.getClassLabel();
	for (var class_index = 0; class_index < this.num_label_classes; ++class_index) {
		this.class_activations[class_index] = (class_index == true_label ? 1 : 0);
	}
};


// Update positive and negative gradients from the given training example. This can be
// called on multiple examples, each adds to the gradients. If the gradients are later
// normalized by the number of examples, the net effect is the mean of all examples.
//
// gibbs_sampling_steps: Markov Chain length for Gibbs sampling (CD-k).
//
models.RBN.prototype.updateGradientsFromExample_ = function(training_example, gibbs_sampling_steps) {
	// The pixels of the training example image represent an unbiased sample of the
	// visible distribution we wish to model. 
	this.setVisibleActivationFromLabeledExample_(training_example);
	
	// Sample the hidden units. This is an unbiased sample of the hidden distribution
	// conditioned on the visible distribution.
	this.sampleHiddenFromVisibleUnits_();
	this.addPhaseToGradients_(1.0);
	// Perform Gibbs sampling to obtain an unbiased sample of the hidden distribution
	// marginalizing over the visible distribution.
	for (var gibbs_step = gibbs_sampling_steps - 1; gibbs_step >= 0; --gibbs_step) {
		is_last = gibbs_step == 0;
		this.sampleVisibleFromHiddenUnits_(is_last);
		this.sampleHiddenFromVisibleUnits_(is_last);
	}
	this.addPhaseToGradients_(-1.0);
};


// Clear accumulated gradients for weights and biases in the model.
//
models.RBN.prototype.clearGradients_ = function() {
	this.weights_gradient.forEach(function(row) {
		row.fill(0.0);
	});
	this.visible_bias_gradient.fill(0.0);
	this.hidden_bias_gradient.fill(0.0);
};


// Compute gradient from current hidden and visible activations.
// The result is added onto existing values in `total_gradient`.
//
models.RBN.prototype.addPhaseToGradients_ = function(factor) {
	// The gradient is the outer product of the visible and hidden activation vectors.
	for (var vi = 0; vi < this.num_visible_units_and_labels; ++vi) {
		for (var hi = 0; hi < this.num_hidden_units; ++hi) {
			// TODO: using one array for accumulating positive and negative
			// phase across a minibatch might be prone to cancellation.
			this.weights_gradient[vi][hi] += factor * this.visible_activations[vi] *
				this.hidden_activations[hi];
		}
	}
	for (var vi = 0; vi < this.num_visible_units_and_labels; ++vi) {
		this.visible_bias_gradient[vi] += factor * this.visible_activations[vi];
	}
	for (var hi = 0; hi < this.num_hidden_units; ++hi) {
		this.hidden_bias_gradient[hi] += factor * this.hidden_activations[hi];
	}
};


// Adjust model weights using gradients.
// scaled_learning_rate: Learning rate scaled by inverse minibatch size.
//
models.RBN.prototype.applyWeightUpdates_ = function(scaled_learning_rate) {
	for (var vi = 0; vi < this.num_visible_units_and_labels; ++vi) {
		for (var hi = 0; hi < this.num_hidden_units; ++hi) {
			this.weights[vi][hi] += scaled_learning_rate * this.weights_gradient[vi][hi];
		}
	}
	for (var vi = 0; vi < this.num_visible_units_and_labels; ++vi) {
		this.visible_bias[vi] += scaled_learning_rate * this.visible_bias_gradient[vi];
	}
	for (var hi = 0; hi < this.num_hidden_units; ++hi) {
		this.hidden_bias[hi] += scaled_learning_rate * this.hidden_bias_gradient[hi];
	}
};


// Sample an activation of the hidden units from the current visible activation.
// use_expected_value: Uses the expected value of each activation instead of a
//    binary activation.
//
models.RBN.prototype.sampleHiddenFromVisibleUnits_ = function(use_expected_value) {
	for (var hi = 0; hi < this.num_hidden_units; ++hi) {
		// The activation probability is f(W*x + b) where f() is the non-linearity (Sigmoid).
		// This is the same rule as for Feed-Forward Neural Nets except we treat the
		// result as a probability from which we sample a (binary) activation.
		var activation_logit = this.hidden_bias[hi];
		for (var vi = 0; vi < this.num_visible_units_and_labels; ++vi) {
			activation_logit += this.weights[vi][hi] * this.visible_activations[vi];
		}
		var activation_probability = util.Sigmoid(activation_logit);
		this.hidden_activations[hi] = use_expected_value
			? activation_probability
			: util.SampleBernoulli(activation_probability);
	}
}



// Sample an activation of the visible units from the current hidden activation.
// use_expected_value: Uses the expected value of each activation instead of a
//    binary activation.
//
models.RBN.prototype.sampleVisibleFromHiddenUnits_ = function(use_expected_value) {
	// Use regular logistic activation for the binary units (same as for
	// sampleHiddenFromVisibleUnits_). 
	for (var vi = 0; vi < this.num_visible_units; ++vi) {
		var activation_logit = this.visible_bias[vi];
		for (var hi = 0; hi < this.num_hidden_units; ++hi) {
			activation_logit += this.weights[vi][hi] * this.hidden_activations[hi];
		}
		var activation_probability = util.Sigmoid(activation_logit);
		this.visible_activations[vi] = use_expected_value
			? activation_probability
			: util.SampleBernoulli(activation_probability);
	}

	// Use Softmax for the K categorical class label units.
	// p(x_i) = exp(x_i) / sum_j=0:K(exp(x_j))
	var class_offset = this.num_visible_units;
	for (var class_index = 0; class_index < this.num_label_classes; ++class_index) {
		var vi = class_offset + class_index;
		var activation_logit = this.visible_bias[vi];
		for (var hi = 0; hi < this.num_hidden_units; ++hi) {
			activation_logit += this.weights[vi][hi] * this.hidden_activations[hi];
		}
		this.class_activations[class_index] = Math.exp(activation_logit);
	}
	var inv_sum = 1.0 / this.class_activations.reduce(util.SumReducer);
	for (var class_index = 0; class_index < this.num_label_classes; ++class_index) {
		this.class_activations[class_index] *= inv_sum;
	}
	if (!use_expected_value) {
		util.SampleCategoricalInPlace(this.class_activations);
	}
};


// Check weights, biases and activations for finiteness. Since hidden and visible
// layers are fully connected with each other, any NaN or Infinity would immediately
// propagate and ruin the net. 
//
models.RBN.prototype.assertFiniteness_ = function() {
	for (var vi = 0; vi < this.num_visible_units; ++vi) {
		console.assert(isFinite(this.visible_bias[vi], 'visible bias is non-finite: ', vi));
		console.assert(isFinite(this.visible_activations[vi]), 'visible unit is non-finite: ', vi); 
		for (var hi = 0; hi < this.num_hidden_units; ++hi) {
			console.assert(isFinite(this.weights[vi][hi]), 'weight is non-finite: ', vi, ' ', hi);
			console.assert(isFinite(this.hidden_activations[hi]), 'hidden unit is non-finite: ', hi); 
		}
	}
	for (var hi = 0; hi < this.num_hidden_units; ++hi) {
		console.assert(isFinite(this.hidden_bias[hi], 'hidden bias is non-finite: ', hi));
	}
};


// Helper to create and initialize a 2D array as array-of-arrays where the inner
// level is composed of typed arrays of `ArrayType`.
//
models.RBN.prototype.create2DArray_ = function(ArrayType, rows, cols, init_func) {
	var arr = new Array(rows);
	for (var i = 0; i < rows; ++i) {	
		arr[i] = new ArrayType(cols);
		if (init_func) {
			for (var j = 0; j < cols; ++j) {
				arr[i][j] = init_func(i, j);
			}
		}
	}
	return arr;
};


// Helper to create and initialize a 1D of type `ArrayType`.
//
models.RBN.prototype.create1DArray_ = function(ArrayType, dim, init_func) {
	var arr = new ArrayType(dim);
	if (init_func) {
		for (var i = 0; i < dim; ++i) {
			arr[i] = init_func(i);
		}
	}
	return arr;
};