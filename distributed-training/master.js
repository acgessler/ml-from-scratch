var dist = dist || {};

// Master for distributed server (Note: distributed means using Web Workers and
// communicating through messages. While not technically distributed, this employs
// the same techniques one would use to spread training across a cluster of machines)
//
// A master wraps around an existing model type, offering the same set of operations.
// It controls a pool of Web Workers, each of which performs performs training on a
// subset of the training data. The master acts as a parameter server for the workers.
// It collects subset gradients from workers, applies them to the model and distributes
// new sets of parameters to all workers.
//
// num_workers: Number of parallel workers to use.
// script_name: Path to the script that implements the model, relative to server root.
// model_class_name: Fully qualified name of the model type, i.e. "models.RBN"
dist.Master = function(num_workers, model_script_name, model_class_name
	/* , extra arguments to model constructor */) {
	this.num_workers = num_workers;
	this.model_script_name = model_script_name;
	this.model_class_name = model_class_name;
	this.workers = [];
	// This is a bit messy: we create n workers. In each worker context, we execute
	// a piece of code that loads all necessary dependencies using importScripts(),
	// then creates a worker instance with the same model parameters as the original.
	// This currently relies on serializing the model arguments into strings, which
	// will work for primitive types only.
	var model_arguments = Array.prototype.slice.call(arguments, 3);
	var string_model_arguments = JSON.stringify(model_arguments).slice(1, -1);
	var preload_scripts = [
		"shared/util.js",
		"shared/example.js",
		"shared/image_example.js",
		"shared/label.js",
		"shared/model.js",
		"distributed-training/worker.js"
	];
	var server_root = document.location.href.replace(/(.*?\/\/.*?\/).*/,'$1');
	var import_scripts = preload_scripts
		.concat([model_script_name])
		.map(function(script_name) { return '"' + server_root + script_name + '"'; })
		.join(',');
	var prelude = 'importScripts(' + import_scripts +');\n';
	for (var i = 0; i < this.num_workers; ++i) {
		var worker_js_text = prelude +
			'var WorkerType = dist.MakeDistributedModelWorkerType(' + model_class_name + ');\n' +
			'var worker = new WorkerType(' + i +', ' + string_model_arguments + ')\n' +
			'worker.startWorker();';
		this.workers.push(this.spawnWorker_(i, worker_js_text));
	}
	// Temporary storage for loaned worker parameters respective parameter updates.
	this.worker_parameters = new Array(this.num_workers);
	this.worker_parameters_update = new Array(this.num_workers);
};


// Spawn the worker with the given index and have it execute the given JS.
//
dist.Master.prototype.spawnWorker_ = function(index, worker_js_text) {
	var blob = new Blob([worker_js_text], {type: 'application/javascript'});
	return new Worker(window.URL.createObjectURL(blob));
};


// Copy the master parameters into the loaned parameters for the given worker.
// This assumes `worker_parameters[worker_index]` is populated.
//
dist.Master.prototype.updateLoanedWorkerParameters_= function(worker_index) {
	var worker_parameters = this.worker_parameters[worker_index];
	if (!worker_parameters) {
		return;
	}
	for (var k in this.parameters) {
		models.Model.copyParameter_(this.parameters[k], worker_parameters[k]);
	}
};


// Return loaned parameters to the given worker. This will not wait for the message to be
// processed by the worker, but the FIFO property of the message queue ensures the
// parameters are applied before the worker processes any other message.
//
dist.Master.prototype.returnLoanedParametersToWorker_ = function(worker_index) {		
	// Transfer all parameters (and previous parameter updates) back to the worker.
	var message = {
		'parameters' : this.worker_parameters[worker_index],
		'parameters_update': this.worker_parameters_update[worker_index]
	};
	this.workers[worker_index].postMessage(message, util.CollectTransferables(message));
	this.worker_parameters[worker_index] = null;
	this.worker_parameters_update[worker_index] = null;
};



// Same as train(), but performs distributed training.
// The master model lets each worker run `num_batches` batches independently. After a batch,
// a worker will send a parameter update to the master and may receive a new set of parameters
// that includes their own update but not necessarily all the other worker's latest updates.
//
// This method relies on the resilience of Stochastic Gradient Descent against delayed
// or partial updates (see also [1] - however, data races are not an issue for JS).
// At worstcase, a batch runs on parameters that are `(num_workers - 1) * num_batches`
// batches behind. This would happen if one worker were starved while the `num_workers - 1`
// other workers complete all their assigned batches, contributing `num_batches` gradient
// updates each.
//
// Returns: A Promise that is fullfilled once training the given number of batches completed.
//
// [1] Hogwild!: A Lock-Free Approach to Parallelizing Stochastic. Gradient Descent. Feng Niu,
// Benjamin Recht, Christopher Ré and Stephen J. Wright.
//
dist.Master.prototype.trainDistributed = function(examples,
	num_batches, batch_size, noshuffle /*, additional training arguments */ ) {
	var self = this;

	var total_updates_received = 0;
	return Promise.all(this.workers.map(function(worker, i) {
		// Sample enough examples for all workers.
		var worker_examples = self.collectExamples_(examples, batch_size * num_batches, noshuffle)
			.map(function(example) {
				return example.serialize();
			});
		// For the first batch, have the worker transfer us their parameters so we can fill
		// in the current state of the model.
		worker.postMessage({
	    	'sync' : true
	    });
		worker.postMessage({
	    	'train' : {
	    		'examples' : worker_examples,
	    		'num_batches' : num_batches,
	    		'batch_size' : batch_size,
	    		'extra_args' : Array.prototype.slice.call(arguments, 4)
	    	}
	    });

		var last_time_worker_parameters_updated = -1;
	    return new Promise(function(resolve, reject) {
	    	var remaining_batches = num_batches;
			worker.onmessage =  function(e) {
				var message = e.data;
				if (!message) {return;}
				// The sender transfered ownership of worker parameters and updates with this
				// message. This saves copying the data each time it send through the message
				// queue (Note: only typed arrays are transferable here, we will still make
				// copies of regular arrays).
				if (message['parameters_update']) {
					self.worker_parameters_update[i] = message['parameters_update'];
					self.applyParameterUpdates_(
						message['scaled_learning_rate'],
						self.worker_parameters_update[i]
					);
					self.assertFiniteness_();
					++total_updates_received;
					--remaining_batches;
				}
				if (message['parameters']) {
		    		self.worker_parameters[i] = message['parameters'];
		    	}

		    	// Only update worker parameters from master if at least half of the other
		    	// workers contributed updates since the last time we sync'd parameters.
		    	// Each worker will still apply their own gradient updates.
		    	if (last_time_worker_parameters_updated == -1 ||
		    		total_updates_received - last_time_worker_parameters_updated > self.num_workers / 2) {
		    		
		    		self.updateLoanedWorkerParameters_(i);
		    		last_time_worker_parameters_updated = total_updates_received;
				}
		    	self.returnLoanedParametersToWorker_(i);
		    	if (remaining_batches == 0) {
		    		resolve();
		    	}
		    };	 
		});
	}));
};


// Variant of trainDistributed(). After each worker has executed a batch, a synchronization point
// is introduced, which accumulates the individual parameter updates and applies them to the master
// model. Then, the updated parameters are sync'd to all the workers. The net effect is as if the
// batch size were scaled by the number of workers.
//
// Returns: A Promise that is fullfilled once training the given number of batches completed.
//
dist.Master.prototype.trainDistributedWithExplicitSynchronization = function(examples,
	num_batches, batch_size, noshuffle /*, additional training arguments */ ) {
	var self = this;

	var all_batches_done = Promise.resolve();
	// Batch boundaries act as a sync point to ensure model replicas stay in sync.
	for (var batch = 0; batch < num_batches; ++batch) {
		var current_batch_done = Promise.all(this.workers.map(function(worker) {
			var worker_examples = self.collectExamples_(examples, batch_size, noshuffle)
    			.map(function(example) {
    				return example.serialize();
    			});
    		// For the first batch, have the worker transfer us their parameters
    		// so we can send back the current state of the model.
    		if (batch == 0) {	
    			worker.postMessage({
    				'sync' : true
    			});
    		}
			worker.postMessage({
		    	'train' : {
		    		'examples' : worker_examples,
		    		'num_batches' : 1,
		    		'batch_size' : batch_size,
		    		'extra_args' : Array.prototype.slice.call(arguments, 4)
	    		}
		    });
			return new Promise(function(resolve, reject) {
				worker.onmessage =  function(e) {
					var message = e.data;
					if (!message) {return;}
					if (message['parameters_update']) {
						self.applyParameterUpdates_(
							message['scaled_learning_rate'],
							message['parameters_update']
						);
					}
		    		self.worker_parameters[i] = message['parameters'];
		    		self.worker_parameters_update[i] = message['parameters_update'];

		    		if (message['parameters_update']) {
		    			resolve();
		    		}
			    };	 
			});
		}));
		all_batches_done = all_batches_done.then(function() {		
			return current_batch_done.then(function() {
				for (var i = 0; i < this.num_workers; ++i) {
					self.updateLoanedWorkerParameters_(i);
		    		self.returnLoanedParametersToWorker_(i);
		    	}
			});
		});
	}
	return all_batches_done;
};


// Given a `models.Model` type, create a new type that represents the same model but
// performs distributed training. The result type mixes in `dist.Master` functionality.
//
// Extra constructor arguments (specified before ModelType arguments): see `dist.Master`.
//
dist.MakeDistributedModelMasterType = function(ModelType) {
	return util.Mixin(ModelType, dist.Master, 3 /* num_workers c'tor argument */);
};
