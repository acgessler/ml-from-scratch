
var util = util || {};

// Sigmoid non-linearity
util.Sigmoid = function(t) {
	return 1.0 / (1.0 + Math.exp(-t));
};

// Logit (inverse Sigmoid, calculates log-odds for a given probability)
util.Logit = function(p) {
	return Math.log(p / (1.0 - p));
};

// Cross entropy supporting scalar and vector inputs.
util.CrossEntropy = function(p, q) {
	p = p.length ? p : [p];
	q = q.length ? q : [q];
	console.assert(p.length == q.length);
	var acc = 0.0;
	for (var i = 0; i < p.length; ++i) {
		acc += p[i] * Math.log(q[i]);
	}
	return -acc;
};

util.RandomInt = function(lower_inclusive, upper_exclusive) {
	return Math.floor(Math.random() * (upper_exclusive - lower_inclusive) + lower_inclusive);
};


util.RandomElement = function(arr) {
	return arr[util.RandomInt(0, arr.length)];
};


util.SampleBernoulli = function(activation_probability) {
	return (Math.random() < activation_probability ? 1 : 0);
};


util.SampleCategoricalInPlace = function(category_probabilities) {
	if (!category_probabilities.length) {
		return;
	}
	var rand = Math.random();
	var selected_category = -1;
	var accum = 0.0;
	for (var i = 0; i < category_probabilities.length; ++i) {
		accum += category_probabilities[i];
		if (rand < accum) {
			selected_category = i;
			break;
		}
	}
	if (selected_category == -1) {
		// This happens if we experience numerical loss when summing the
		// categories. In this case, we just sample a random category.
		selected_category = util.RandomInt(0, category_probabilities.length);
	}
	category_probabilities.fill(0.0);
	category_probabilities[selected_category] = 1.0;
};

util.SumReducer = function(a,b) {return (a || 0) + (b || 0);};

util.Mean = function(arr) {
	if (!arr.length) {
		return NaN;
	}
	return arr.reduce(util.SumReducer) / arr.length;
};

util.SampleNormal = /* function(mean, variance) */ (function() {
	// Straightforward JS version of the C code in Wikipedia / Box Mueller Transform
	// see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
	var two_pi = 2.0*3.14159265358979323846;
	// Minimum double value that will still be stored in normalized form. Below this
	// value, Math.log() might yield -INF. 
	var epsilon =  2.2251e-308; 

	var z0, z1;
	var generate = true;

	return function(mean, variance) {
		mean = mean || 0.0;
		variance = variance || 1.0;
		if (!generate) {
			generate = true;
			return z1 * variance + mean;
		}

		var u1, u2;
		do
		{
			u1 = Math.random();
		}
		while (u1 <= epsilon);

		u2 = Math.random();
		z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(two_pi * u2);
		z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(two_pi * u2);

		// Return z0 this time and z1 next time (z0 and z1 are i.i.d).
		generate = false;
		return z0 * variance + mean;
	};
})();

util.IsTypedArray = function(arr) {
	return typeof arr.BYTES_PER_ELEMENT != 'undefined';
};

util.IsArrayOrTypedArray = function(arr) {
	return Array.isArray(arr) || util.IsTypedArray(arr);
};

util.TypedArrayReplacer = function(key, value) {
	if (!util.IsTypedArray(value)) {
		return value;
	}
	return Array.from(value);
};