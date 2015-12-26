
var io = io || {};

// Represents an integer class label.
// Strong labels must be preprocessed with a dictionary that maps them to (dense)
// integer indices.
//
io.Label = function(class_label) {
	this.class_label = class_label;
};


// Gets the integer class label.
//
io.Label.prototype.getClassLabel = function() {
	return this.class_label;
};


// Given an `ExampleType` (which must inherit from io.Example), produces a new
// class type that also provides a label. All methods of io.Label are available
// on the created type. The constructor first processes io.Label parameters and
// forwards the remainder to the `ExampleType` constructor.
//
io.MakeLabeledExampleType = function(ExampleType) {
	var TransformedExampleType = function(class_label /* ,arguments */) {
		io.Label.call(this, class_label);
		ExampleType.apply(this, Array.prototype.slice.call(arguments, 1));
	};
	for (var k in ExampleType.prototype) {
		TransformedExampleType.prototype[k] = ExampleType.prototype[k];
	}
	for (var k in io.Label.prototype) {
		TransformedExampleType.prototype[k] = io.Label.prototype[k];
	}
	TransformedExampleType.prototype.constructor = TransformedExampleType;
	return TransformedExampleType;
};

// Labeled example types for all example types.
io.LabeledExample = io.MakeLabeledExampleType(io.Example);
io.LabeledImageExample = io.MakeLabeledExampleType(io.ImageExample);
