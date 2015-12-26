
mnist_reader = {};

mnist_reader.MNIST_WIDTH = 28;
mnist_reader.MNIST_HEIGHT = 28;

mnist_reader.LABEL_HEADER = 8;
mnist_reader.IMAGES_HEADER = 15;

// Read a MNIST data set (train or test) from the given buffers.
// data_buffer: Buffer containing images.
// label_buffer: Buffer containing labels.
// Returns an array of io.LabeledImageExample objects.
//
mnist_reader.readLabeledImageExamplesFromBuffer = function(data_buffer, label_buffer) {
	var labeled_examples = [];
	var data_cursor = mnist_reader.IMAGES_HEADER;
	var label_cursor = mnist_reader.LABEL_HEADER;
	while (data_cursor <= data_buffer.length && label_cursor <= label_buffer.length) {
		var pixels = new Uint8ClampedArray(mnist_reader.MNIST_WIDTH * mnist_reader.MNIST_HEIGHT);
		var pixel_cursor = 0;
		for (var x = 0; x < mnist_reader.MNIST_WIDTH; ++x) {
	        for (var y = 0; y < mnist_reader.MNIST_HEIGHT; ++y) {
	        	pixels[pixel_cursor++] = data_buffer[data_cursor++];
	        }
	    }
	    var class_label = label_buffer[label_cursor++];
	    var labeled_example = new io.LabeledImageExample(
	    	class_label,
	    	pixels,
	    	mnist_reader.MNIST_WIDTH,
	    	mnist_reader.MNIST_HEIGHT,
	    	/* components */ 1);
	    labeled_examples.push(labeled_example);
	}
	return labeled_examples;
};


// Asynchronously loads MNIST train and test sets from a HTTP location.
// url_prefix: prefix of the URL where the MNIST data is located. This assumes standard
//   (uncomressed) filenames:
//      <prefix>train-images-idx3-ubyte
//      <prefix>train-labels-idx1-ubyte
//      <prefix>t10k-images-idx3-ubyte
//      <prefix>t10k-labels-idx1-ubyte
// callback: Function that is invoked when loading is complete.
//    Parameters: two arrays of io.LabeledImageExample objects representing the training
//      and test MNIST data sets, respectively. No parameters if an error occurs.
//    Return value: ignored.
//
mnist_reader.loadTrainAndTestSetsViaXhr = function(url_prefix, callback) {
	var request_state = [
		[url_prefix + 'train-images.idx3-ubyte', null],
		[url_prefix + 'train-labels.idx1-ubyte', null],
		[url_prefix + 't10k-images.idx3-ubyte',  null],
		[url_prefix + 't10k-labels.idx1-ubyte',  null],
	];
	var countdown = request_state.length;
	var cancelled = false;
	request_state.forEach(function(entry) {
		var xhr = new XMLHttpRequest();
		xhr.addEventListener("load", function() {
			if (cancelled) {
				return;
			}
			if (!xhr.response) {
				cancelled = true;
				callback();
			}
			entry[1] = xhr.response;
			if (--countdown == 0) {
				var training = mnist_reader.readLabeledImageExamplesFromBuffer(
					new Uint8Array(request_state[0][1]),
					new Uint8Array(request_state[1][1]));
				var test = mnist_reader.readLabeledImageExamplesFromBuffer(
					new Uint8Array(request_state[2][1]),
					new Uint8Array(request_state[3][1]));
				callback(training, test);
			}
		});
		xhr.responseType = "arraybuffer";
		xhr.open("GET", entry[0]);
		xhr.send();
	});
};
