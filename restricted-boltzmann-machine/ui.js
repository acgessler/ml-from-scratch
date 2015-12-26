// Main controller logic for the (browser-based) RBN / MNIST sample.

var rbn;
var train_examples;
var test_examples;
var train_running = false;
var batches = 0;

var HIDDEN_UNITS = 1000;

var FIXED_LEARNING_RATE = 0.01;
var AUTO_TUNE_LEARNING_RATE = true;
var learning_rate = -1;

// Array-typed stats are averaged over multile values to reduce noise.
var STATS_AVERAGING_WINDOW = 10;
var approx_eval_classification_error = [];
var approx_eval_reconstruction_error = [];
// Evaluation error on training set is for diagnosing overfitting.
var approx_eval_classification_error_train = [];
var approx_eval_reconstruction_error_train = [];
var full_eval_classification_error = 1.0;
var full_eval_reconstruction_error = 100000;

var BATCH_SIZE = 20;
var EXAMPLE_SAMPLES = 30;
var RECON_SAMPLES = 10;

var EVAL_FREQUENCY = 25;
var UPDATE_RECONSTRUCTION_FREQUENCY = 25;
var UPDATE_FILTERS_FREQUENCY = 10;
var UPDATE_SAMPLES_FREQUENCY = 10;
// Number of samples for approximate eval.
var EVAL_SIZE = 150;

function Init() {
	var pixel_count = mnist_reader.MNIST_WIDTH * mnist_reader.MNIST_HEIGHT;
	rbn = new models.RBN(pixel_count, HIDDEN_UNITS, 10);

	UpdateSampleTrainingExamples();
	UpdateSampleTestExamples();
	UpdateFilters();
	UpdateReconstructionExamples();
	UpdateStats();
}

function GetTemporaryDigitSizedCanvas() {
	return $('#temp_digit_canvas')[0];
}

function UpdateSampleTrainingExamples() {
	UpdateSamples($("#sample_training_examples"), train_examples);
}

function UpdateSampleTestExamples() {
	UpdateSamples($("#sample_test_examples"), test_examples);
}

function UpdateSamples($container, examples) {
	$container.empty();
	var entries = [];
	for (var i = 0; i < EXAMPLE_SAMPLES; ++i) {
		example = util.RandomElement(examples);
		entries.push([example, rbn.classifyExample(example)]);
	}
	var $row = $('<div>');
	entries.forEach(function(entry) {
		var img = new Image();
		entry[0].getImage(img, GetTemporaryDigitSizedCanvas());
		$row.append(img);
	});
	$container.append($row);

	$row = $('<div>');
	entries.forEach(function(entry) {
		$row.append('<div class="label">' + entry[0].getClassLabel() + "</div>");
	});
	$container.append($row);
	$row = $('<div>');
	entries.forEach(function(entry) {
		var is_correct = entry[1] == entry[0].getClassLabel() ;
		$row.append('<div class="label ' + (is_correct ? ' correct' : ' incorrect') + '">' + entry[1] + '</div>');
	});
	$container.append($row);
}

function UpdateReconstructionExamples() {
	$("#sample_recon_examples").empty();
	for (var digit = 0; digit < 10; ++digit) {
		var row = $('<div>');
		row.append('<div class="label">' + digit + '</div>');
		
		for (var i = 0; i < RECON_SAMPLES; ++i) {
			// Create an example that is all zero, but sets the label index.
			var noise_in = new Uint8Array(
				mnist_reader.MNIST_WIDTH * mnist_reader.MNIST_HEIGHT);
			//for (var j = 0; j < noise_in.length; ++j) {
			//	noise_in[j] = 127 + Math.random() * 5.0;
			//}
			var example = new io.LabeledImageExample(digit, noise_in,
				mnist_reader.MNIST_WIDTH,
				mnist_reader.MNIST_HEIGHT);
			var pixels = rbn.reconstructVisibleUnitsForExample(example);
			for (var j = 0; j < pixels.length; ++j) {
				pixels[j] = pixels[j] * 255.0;
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
		'approx_eval_classification_error: ' + util.Mean(approx_eval_classification_error) + '\n' +
		'approx_eval_reconstruction_error: ' + util.Mean(approx_eval_reconstruction_error) + '\n' +
		'approx_eval_classification_error_train: ' + util.Mean(approx_eval_classification_error_train) + '\n' +
		'approx_eval_reconstruction_error_train: ' + util.Mean(approx_eval_reconstruction_error_train) + '\n' +
		'full_eval_classification_error: ' + full_eval_classification_error + '\n' +
		'full_eval_reconstruction_error: ' + full_eval_reconstruction_error + '\n'
	);
	$('#training_log').text(
		'batches: ' + batches + '\n' +
		'total_examples: ' + batches * BATCH_SIZE + '\n' +
		'learning_rate: ' + learning_rate + '\n' +
		'batch_size: ' + BATCH_SIZE + '\n'
	);
}

function TrainSingleBatch(update_all) {
	++batches;
	console.log('BATCH ' + batches);
	learning_rate = rbn.train(train_examples, 1, BATCH_SIZE, 1,
		AUTO_TUNE_LEARNING_RATE ? -1 : FIXED_LEARNING_RATE);
	console.log('train done');
	
	if (update_all || batches % UPDATE_SAMPLES_FREQUENCY == 0) {
		UpdateSampleTrainingExamples();
		UpdateSampleTestExamples();
		console.log('updatesamples done');
	}

	if (update_all || batches % UPDATE_FILTERS_FREQUENCY == 0) {
		UpdateFilters();
		console.log('updatefilters done');
	}

	if (update_all || batches % UPDATE_RECONSTRUCTION_FREQUENCY == 0) {
		UpdateReconstructionExamples();
		console.log('addreconexamples done');
	}

	if (update_all || batches % EVAL_FREQUENCY == 0) {
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
	train_running = false;
}

function AddStat(arr, stat) {
	arr.push(stat);
	if (arr.length > STATS_AVERAGING_WINDOW) {
		arr.shift();
	}
}

function EvalApproximate() {
	AddStat(approx_eval_classification_error, rbn.evalClassificationError(test_examples, EVAL_SIZE));
	AddStat(approx_eval_reconstruction_error, rbn.evalReconstructionError(test_examples, EVAL_SIZE));
	AddStat(approx_eval_classification_error_train, rbn.evalClassificationError(train_examples, EVAL_SIZE));
	AddStat(approx_eval_reconstruction_error_train, rbn.evalReconstructionError(train_examples, EVAL_SIZE));
}

function EvalFull() {
	full_eval_classification_error = rbn.evalClassificationError(test_examples_, /* no sampling */ -1);
	full_eval_reconstruction_error = rbn.evalReconstructionError(test_examples_, /* no sampling */ -1);
	UpdateStats();
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
    mnist_reader.loadTrainAndTestSetsViaXhr('../data/mnist/', function(train_examples_, test_examples_) {
    	train_examples = train_examples_;
    	test_examples = test_examples_;
    	console.log("Loaded training examples: " + train_examples.length);
    	console.log("Loaded test examples: " + test_examples.length);
    	Init();
    });

    $('#train_batch').click(function() {TrainSingleBatch(true); });
    $('#train_start').click(TrainStart);
    $('#train_stop').click(TrainStop);
    $('#full_eval_test').click(EvalFull);
});