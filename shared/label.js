
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
// class type that also provides a label. All methods of `io.Label` are available
// on the created type. The constructor first processes `io.Label` parameters and
// forwards the remainder to the `ExampleType` constructor.
//
io.DeclareLabeledExampleType = function(name, ExampleType) {
	var LabelType = util.Mixin(ExampleType, io.Label, 1);
	LabelType.prototype.typename = name;
	util.Declare(name, LabelType);
};

// Labeled example types for all example types.
io.DeclareLabeledExampleType('io.LabeledExample', io.Example);
io.DeclareLabeledExampleType('io.LabeledImageExample', io.ImageExample);
