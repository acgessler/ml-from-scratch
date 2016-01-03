
var io = io || {};

// Base example that consists of a set of (primitive-valued) features.
//
// features: Raw array-like object of features. Read-only. Example retains
//   a reference to it. Values can be continuous and of any range.
io.Example = function(features) {
	this.features = features;
};

io.Example.prototype.typename = "io.Example";


// Serialize into a JSON-ish form.
// Saves all simple so most sub classes should not need to change it.
//
io.Example.prototype.serialize = function() {
	var data = {};
	for (var k in this) {
		if (typeof this[k] !== 'function') {
			data[k] = this[k];
		}	
	}
	return data;
};


// Serialize from a JSON-ish form (non-static, presumes class type is known).
// Restores all simple fields so most sub classes should not need to change it.
//
io.Example.prototype.deserializeFrom = function(data) {
	for (var k in data) {
		this[k] = data[k];
	}
};



// Deserialize from a JSON-ish form.
// Static method, returns the appropriate class type. Restores all simple fields
//
io.Example.deserialize = function(data) {
	var ExampleType = eval(data.typename);
	if (!ExampleType) {
		return null;
	}
	var example = Object.create(ExampleType.prototype);
	example.deserializeFrom(data);
	return example;
};


// Extracts the continuous (raw) features of the example.
//
// Values can be continuous and of any range. If a model required scaled
// (i.e., range [0,1]) or normalized (i.e., L2) features, it would either
// preprocess all examples or convert them to the desired format when needed.
//
io.Example.prototype.getContinuousFeatures = function() {
	return this.features;
};


// Extract the features as binary features where any value above the mean
// feature value counts as "1" and any value below the mean feature value
// as "0".
//
io.Example.prototype.getBinaryFeatures = function() {
	var mean = this.features.reduce(util.SumReducer) / this.features.length;
	return this.features.map(function(feature) {
		return feature > mean ? 1 : 0;
	});
};

