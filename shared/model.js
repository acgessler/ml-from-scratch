var models = models || {};

// Base class for models.
//
// name: Unique name to identify the kind of model.
// parameters: Dictionary with references to all model parameters that require
//   to be serialized/deserialized. All values are expected to be Arrays of either
//   Arrays or Float32Arrays, or Float32Arrays.
// parameters_update: A matching dictionary with the same topology as `parameters`.
//   Leaf arrays will hold parameter updates (gradients).
// hyperparameters: Dictionary containing (static) hyper parameters for the
//   model. These will, in most cases, not change during the lifetime of an instance.
//   In some case thay are not known yet at construction time (i.e. learning rate)
//   but are set when training starts instead. If omitted, an empty object is set.
//
models.Model = function(name, parameters, parameters_update, hyperparameters) {
	this.name = name;
	this.parameters = parameters;
	this.parameters_update = parameters_update;
	this.hyperparameters = hyperparameters || {};

	this.collect_examples_cursor = 0;
};


// Train on the given labeled data set. This defines shared parameters for train().
// Deriving classes may add more model-specific parameters to the end of the
// argument list, but are asked to provide defaults.
//
// examples: Examples to train on.
// num_batches: Number of training batches.
// batch_size: Number of (randomly chosen) examples per batch, defaults to 1.
// noshuffle: If true, drawn sequentially instead of randomly. Defaults to false.
//
models.Model.prototype.train = function(examples,  num_batches, batch_size, noshuffle
	/* additional training parameters */) {
};


// Serialize model state into a state object that can be serialized to JSON.
//
models.Model.prototype.serialize = function() {
	return {
		name: this.name,
		parameters: this.parameters,
		hyperparameters: this.hyperparameters,
	};
};


// De-serializes the given state object into the model.
// This is not necessarily a full update, any part of the model state which is not
// specified in the input state object is left unchanged.
//
// Returns true if no error occured. False indicates incompatibility of the serialized
// state with the model. Currently, the error can only be found in the console.
//
models.Model.prototype.deserialize = function(state) {
	state = this.convertSerializedState_(state);
	
	if (state.name != this.name) {
		console.log('Serialized state has mismatching model name: ' +
			', my value '  + this.name +
			', new value ' + state.name);
		return false;
	}
	// Update hyper parameters. By default this will reject any changes. 
	for (var k in state.hyperparameters) {
		if (!this.isCompatibleHyperParameter_(k, state.hyperparameters[k])) {
			console.log('Hyper parameter is incompatible: ' + k +
				', my value '  + this.hyperparameters[k] +
				', new value ' + state.hyperparameters[k]);
			return false;
		}
		this.hyperparameters[k] = state.hyperparameters[k];
	}
	// Update existing parameter storage in-place. Ignore any parameters that
	// we do not recognize.
	for (var k in state.parameters) {
		if (!this.parameters[k]) {
			console.log('Ignoring unknown parameter: ' + k);
			continue;
		}
		if (!models.Model.copyParameter_(state.parameters[k], this.parameters[k])) {
			return false;
		}
	}
	return true;
};


// Get a given number of examples from a set of examples. By default, examples are
// sampled randomly. This is a utility for deriving classes.
//
// examples: Array of all examples.
// count: Number of examples to retrieve.
// noshuffle: If this parameter is true, examples are not drawn randomly. Instead,
//   the next `count` elements according to a (static) cursor are picked and the
//   cursor is incremented by `count` (treating the examples as a ring).
models.Model.prototype.collectExamples_ = function(examples, count, noshuffle) {
	var arr = new Array(count);
	if (noshuffle) {
		for (var i = 0; i < count; ++i, ++this.collect_examples_cursor) {
			arr[i] = examples[this.collect_examples_cursor % examples.length];
		}
	}
	else {
		for (var i = 0; i < count; ++i) {
			arr[i] = util.RandomElement(examples);
		} 
	}
	return arr;
};


// Given `state` that may have been written by a *foreign* model implementation,
// match up entries as to produce an input compatible with the *this* model.
//
// To be implemented by deriving classes.
//
models.Model.prototype.convertSerializedState_ = function(state) {
	// TODO(acgessler): Can we use this to allow RBMs to be fed into FFNNSs?
	return state;
};


// Checks if a hyper parameter can be changed.
// name: Name of the hyper parameter.
// new_value: Updated value.
// 
// To be implemented by deriving classes. The default implementation will not allow
// any hyper parameter values except for values that compare equal to the current
// values.
//
models.Model.prototype.isCompatibleHyperParameter_ = function(name, new_value) {
	return typeof this.hyperparameters[name] == 'undefined' || this.hyperparameters[name] == new_value;
};


// Adjust model parameters given parameter updates.
// scaled_learning_rate: Effective learning rate to use for the weights update.
// parameters_update: Dictionary of parameter updates (must have matching keys in
// 	`this.parameters`). Each value is a possibly nested array type. The topology
//  of the update tensor must match the model parameters.
//  This can be omitted, in this case `this.parameters_update` is used.
//
models.Model.prototype.applyParameterUpdates_ = function(scaled_learning_rate, parameters_update) {
	parameters_update = parameters_update || this.parameters_update;
	var apply_update = function(data, update) {
		// Leaves are always typed arrays.
		if (util.IsTypedArray(data)) {
			for (var i = 0; i < data.length; ++i) {
				data[i] += scaled_learning_rate * update[i];

			}
		}
		// Regular arrays are for nested data.
		else {
			console.assert(Array.isArray(data) && Array.isArray(update));
			for (var i = 0; i < data.length; ++i) {
				apply_update(data[i], update[i]);
			}
		}
	};

	for (var k in parameters_update) {
		apply_update(this[k], parameters_update[k]);
	}
};


// Copies a parameter from `src` to `dest` component-wise. This retains all array or
// typed array objects in the destination. Each parameter is a possibly nested array
// type. The leaf types are typically typed arrays (i.e., Float32Array).
//
// This is a static method.
// Returns false if the topologies of `src` and `dest` are mismatched.
//
models.Model.copyParameter_ = function(src, dest) {
	if (src.length != dest.length) {
		console.log('Parameter length mismatch: ' + src.length + ' vs. ' + dest.length);
		return false;
	}
	// Regular arrays are for nested data.
	if (Array.isArray(dest)) {
		for (var i = 0; i < dest.length; ++i) {
			if (!models.Model.copyParameter_(src[i], dest[i])) {
				return false;
			}
		}
	}
	// Leaves are always typed arrays.
	else {
		console.assert(util.IsTypedArray(dest), "Leaf arrays must be typed.");
		dest.set(src);
	}
	return true;
};


// Helper to create and initialize a 2D array (as array-of-arrays) where the inner
// level is composed of typed arrays of `TypedArrayType`.
//
models.Model.prototype.create2DArray_ = function(TypedArrayType, rows, cols, init_func) {
	var arr = new Array(rows);
	for (var i = 0; i < rows; ++i) {	
		arr[i] = new TypedArrayType(cols);
		console.assert(util.IsTypedArray(arr[i]));
		if (init_func) {
			for (var j = 0; j < cols; ++j) {
				arr[i][j] = init_func(i, j);
			}
		}
	}
	return arr;
};


// Helper to create and initialize a 1D of type `TypedArrayType`.
//
models.Model.prototype.create1DArray_ = function(TypedArrayType, dim, init_func) {
	var arr = new TypedArrayType(dim);
	console.assert(util.IsTypedArray(arr));
	if (init_func) {
		for (var i = 0; i < dim; ++i) {
			arr[i] = init_func(i);
		}
	}
	return arr;
};
