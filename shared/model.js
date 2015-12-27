var models = models || {};

// Base class for models.
//
// name: Unique name to identify the kind of model.
// parameters: Dictionary with references to all model parameters that require
//   to be serialized/deserialized. All values are expected to be Arrays of either
//   Arrays or Float32Arrays, or Float32Arrays.
// hyperparameters: Dictionary containing (static) hyper parameters for the
//   model. These will, in most cases, not change during the lifetime of an instance.
//   In some case thay are not known yet at construction time (i.e. learning rate)
//   but are set when training starts instead. If omitted, an empty object is set.
//
models.Model = function(name, parameters, hyperparameters) {
	this.name = name;
	this.parameters = parameters;
	this.hyperparameters = hyperparameters || {};
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


// Given `state` that may have been written by a *foreign* model implementation,
// match up entries as to produce an input compatible with the *this* model.
//
// To be implemented by deriving classes.
//
models.Model.prototype.convertSerializedState_ = function(state) {
	// TODO(acgessler): Can we use this to allow RBMs to be fed into FFNNSs?
	return state;
};


// Check if a hyper parameter can be changed.
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


// De-serialize the given state object into the model.
// This is not necessarily a full update, any part of the model state which is not
// specified in the input state object is left unchanged.
//
// Returns true if no error occured. False indicates incompatibility of the serialized
// state with the model. Currently, the error can only be found in the console.
//
models.Model.prototype.deserialize = function(state) {
	state = this.convertSerializedState_(state);
	// Recursively map Array[float] to Float32Array. Keep Array[Array].
	var copy_parameter = function(src, dest) {
		if (src.length != dest.length) {
			console.log('Parameter length mismatch: ' + src.length + ' vs. ' + dest.length);
			return false;
		}
		if (util.IsArrayOrTypedArray(dest[0])) {
			// Nested array.
			for (var i = 0; i < dest.length; ++i) {
				if (!copy_parameter(src[i], dest[i])) {
					return false;
				}
			}
		}
		else {
			dest.set(src);
		}
		return true;
	};
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
		if (!copy_parameter(state.parameters[k], this.parameters[k])) {
			return false;
		}
	}
	return true;
};
