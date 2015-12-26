// Main controller logic for the (browser-based) RBN / MNIST sample.

var rbn;
var train_examples;
var test_examples;
var train_running = false;
var batches = 0;
var learning_rate = 0.01;
var approx_eval_classification_error = 1.0;
var approx_eval_reconstruction_error = 100000;
// Evaluation error on training set is for diagnosing overfitting.
var approx_eval_classification_error_train = 1.0;
var approx_eval_reconstruction_error_train = 100000;
var full_eval_classification_error = 1.0;
var full_eval_reconstruction_error = 100000;

var BATCH_SIZE = 10;
var EXAMPLE_SAMPLES = 20;
var RECON_SAMPLES = 5;

var EVAL_FREQUENCY = 100;
var UPDATE_RECONSTRUCTION_FREQUENCY = 25;
var UPDATE_FILTERS_FREQUENCY = 5;

function Init() {
	var pixel_count = mnist_reader.MNIST_WIDTH * mnist_reader.MNIST_HEIGHT;
	rbn = new models.RBN(pixel_count, Math.floor(pixel_count * 0.6), 10);

	AddSampleTrainingExamples();
	AddSampleTestExamples();
}

function GetTemporaryDigitSizedCanvas() {
	return $('#temp_digit_canvas')[0];
}

function AddSampleTrainingExamples() {
	for (var i = 0; i < EXAMPLE_SAMPLES; ++i) {
		var img = new Image();
		var training_example = util.RandomElement(train_examples);
		training_example.getImage(img, GetTemporaryDigitSizedCanvas());
		$("#sample_training_examples").append(img);
	}
}

function AddSampleTestExamples() {
	for (var i = 0; i < EXAMPLE_SAMPLES; ++i) {
		var img = new Image();
		var test_example = util.RandomElement(test_examples);
		test_example.getImage(img, GetTemporaryDigitSizedCanvas());
		$("#sample_test_examples").append(img);
	}
}

function UpdateReconstructionExamples() {
	$("#sample_recon_examples").empty();
	for (var digit = 0; digit < 10; ++digit) {
		var row = $('<div>');
		row.append('<span>' + digit + ':&nbsp;&nbsp;</span>');
		
		for (var i = 0; i < RECON_SAMPLES; ++i) {
			// Create an example that is all zero, but sets the label index.
			var example = new io.LabeledImageExample(digit, new Uint8Array(
				mnist_reader.MNIST_WIDTH * mnist_reader.MNIST_HEIGHT),
				mnist_reader.MNIST_WIDTH,
				mnist_reader.MNIST_HEIGHT);
			var pixels = rbn.reconstructVisibleUnitsForExample(example);
			for (var j = 0; j < pixels.length; ++j) {
				pixels[j] *= 256.0;
			}
			example.features = pixels; // TODO(acgessler): cleanup.
			var img = new Image();
			example.getImage(img, GetTemporaryDigitSizedCanvas());
			row.append(img);
			
		}
		$("#sample_recon_examples").append(row);
	}
}

function UpdateStats() {
	$('#evaluation_log').text(
		'approx_eval_classification_error: ' + approx_eval_classification_error + '\n' +
		'approx_eval_reconstruction_error: ' + approx_eval_reconstruction_error + '\n' +
		'approx_eval_classification_error_train: ' + approx_eval_classification_error_train + '\n' +
		'approx_eval_classification_error_train: ' + approx_eval_reconstruction_error_train + '\n' +
		'full_eval_classification_error: ' + full_eval_classification_error + '\n' +
		'fulleval_reconstruction_error: ' + full_eval_reconstruction_error + '\n'
	);
	$('#training_log').text(
		'batches: ' + batches + '\n' +
		'total_examples: ' + batches * BATCH_SIZE + '\n' +
		'learning_rate: ' + learning_rate + '\n'
	);
}

function TrainSingleBatch() {
	++batches;
	console.log('BATCH ' + batches);
	rbn.train(train_examples, 1, BATCH_SIZE, 1, 0.01);
	console.log('train done');
	
	if (batches % UPDATE_FILTERS_FREQUENCY == 0) {
		UpdateFilters();
		console.log('updatefilters done');
	}

	if (batches % UPDATE_RECONSTRUCTION_FREQUENCY == 0) {
		UpdateReconstructionExamples();
		console.log('addreconexamples done');
	}

	if (batches % EVAL_FREQUENCY == 0) {
		EvalApproximate();
		console.log('evalapproximate done');
	}
	UpdateStats();
}

function TrainStart() {
	if (train_running) {
		return;
	}
	train_running = true;
	// Schedule batches with a timeout in between to allow browser event processing.
	var continuation = function() {
		if (!train_running) {
			return;
		}
		TrainSingleBatch();
		setTimeout(continuation, 20);
	};
	continuation();
}

function TrainStop() {
	train_stopped = true;
}

function EvalApproximate() {
	var EVAL_SIZE = 100;
	approx_eval_classification_error = rbn.evalClassificationError(test_examples, EVAL_SIZE);
	approx_eval_reconstruction_error = rbn.evalReconstructionError(test_examples, EVAL_SIZE);
	approx_eval_classification_error_train = rbn.evalClassificationError(train_examples, EVAL_SIZE);
	approx_eval_reconstruction_error_train = rbn.evalReconstructionError(train_examples, EVAL_SIZE);
}

function EvalFull() {
	full_eval_classification_error = rbn.evalClassificationError(test_examples_, /* no sampling */ -1);
	full_eval_reconstruction_error = rbn.evalReconstructionError(test_examples_, /* no sampling */ -1);
}

function UpdateFilters() {
	$("#filters").empty();
	canvas = $('#temp_digit_canvas')[0];
	canvas.width = mnist_reader.MNIST_WIDTH ;
	canvas.height = mnist_reader.MNIST_HEIGHT;
	for (var hi = 0; hi < rbn.num_hidden_units; ++hi) {
		var filter = rbn.getFilterForHiddenUnit(hi);
		
		var rgba_triples = new Uint8ClampedArray(mnist_reader.MNIST_WIDTH * mnist_reader.MNIST_HEIGHT * 4);
		for (var src_cursor = 0, dest_cursor = 0; src_cursor < filter.length; ++src_cursor) {
			var val = 256.0 * filter[src_cursor];
			rgba_triples[dest_cursor++] = val;
			rgba_triples[dest_cursor++] = val;
			rgba_triples[dest_cursor++] = val;
			rgba_triples[dest_cursor++] = 255;
		}
		var ctx = canvas.getContext('2d');
		ctx.putImageData(new ImageData(
			rgba_triples,
			mnist_reader.MNIST_WIDTH,
			mnist_reader.MNIST_HEIGHT), 0, 0);

		var img = new Image();
		img.src = canvas.toDataURL();
		$("#filters").append(img);
	}
}

$(document).ready(function(){
    mnist_reader.loadTrainAndTestSetsViaXhr('mnist/', function(train_examples_, test_examples_) {
    	train_examples = train_examples_;
    	test_examples = test_examples_;
    	console.log("Loaded training examples: " + train_examples.length);
    	console.log("Loaded test examples: " + test_examples.length);
    	Init();
    });

    $('#train_batch').click(TrainSingleBatch);
    $('#train_start').click(TrainStart);
    $('#train_stop').click(TrainStop);
});