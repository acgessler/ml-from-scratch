var dist = dist || {};

// Worker mixin for models.
//
// This adds functionality to listen to messages from the training master (master.js).
// Each worker performs independent training, but sends parameter updates (gradients)
// to the master and periodically receives a new copy of the model parameters.
dist.Worker = function(worker_index) {
	this.worker_index = worker_index;
	this.train_done = Promise.resolve();
	this.started = false;
};


// Overrides models.Model.applyParameterUpdates_().
//
dist.Worker.prototype.applyParameterUpdates_ = function(scaled_learning_rate, parameters_update) {
	parameters_update = parameters_update || this.parameters_update;
	// Update this model as usual.
	this._callOriginal();
	// Assemble a message to later send this as an update to the master.
	this.loan_message = {
		'parameters_update' : parameters_update,
		'parameters' : this.parameters,
		'scaled_learning_rate' : scaled_learning_rate
	};
};


// Start listening to messages from the master.
//
dist.Worker.prototype.startWorker = function() {
	console.assert(!this.started, 'startWorker() called twice');
	console.log('Start worker ' + this.worker_index);
	this.started = true;
	var self = this;

	onmessage = function (e) {
		var message = e.data;
		// Register any returned parameter loans.
		if (message['parameters']) {
			self.parameters = message['parameters'];
			// Restore references to transferred objects (these seem to become all NaN with Chrome 47+)
			for (var k in self.parameters) {
				if (typeof self[k] !== 'undefined') {
					self[k] = self.parameters[k];
				}
			}
			
			// TODO: migrate assertFiniteness() into Model base class.
			self.assertFiniteness_();
		}
		if (message['parameters_update']) {
			self.parameters_update = message['parameters_update'];			
			// Restore references to transfered objects (these seem to become all NaN with Chrome 47+)
			for (var k in self.parameters_update) {
				if (typeof self[k] !== 'undefined') {
					self[k + '_gradient'] = self.parameters_update[k];
				}
			}
		}
		if (self.call_when_parameter_loan_returned && self.parameters_update && self.parameters) {
			self.call_when_parameter_loan_returned();
			self.call_when_parameter_loan_returned = null;
		}
		if (message['sync']) {
			var response = {
				'parameters' : self.parameters
			};
			self.parameters = null;
			postMessage(response, util.CollectTransferables(response));
		}
		if (message['train']) {
			var train = message['train'];
			var examples = train['examples'].map(io.Example.deserialize);
			// Wrap the training loop into a continuation to allow the worker to receive and
			// process messages between batches. Each iteration calls train() with the
			// original parameters but a batch count of 1.
			var batch_size = train['batch_size'];
			for (var i = 0; i < train['num_batches']; ++i) {
				(function(batch_examples) {
					self.train_done = self.train_done.then(function() {
						return new Promise(function(resolve) {
							var train_args = [batch_examples, /* num_batches */ 1, batch_size, /* noshuffle */ true];
							self.train.apply(self, train_args.concat(train['extra_args']));
							// Send our parameters to the master and continue once we get them back.
							// The master may or may not decide to update our parameters. It may also
							// decide whether to use our parameter update wholly, in parts or not at
							// all.
							postMessage(self.loan_message, util.CollectTransferables(self.loan_message));
							self.parameters = null;
							self.parameters_update = null;
							self.call_when_parameter_loan_returned = resolve;
						});
					});
				})(examples.slice(i * batch_size, (i + 1) * batch_size));
			}
		}
	};
};


// Given a `models.Model` type, create a new type that represents the same model but
// acts as a worker for distributed trainer. The result type mixes in `dist.Worker`
// functionality.
//
dist.MakeDistributedModelWorkerType = function(ModelType) {
	return util.Mixin(ModelType, dist.Worker, 1);
};
