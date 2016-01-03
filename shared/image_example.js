
var io = io || {};

// Example representing an (2D) image.
//
// Each (sub)pixel is represented using an 8 bit integer value (0..255).
// components: Number of color components, 1 represents a grayscale or monochrome
//   image. Defaults to 1.
io.ImageExample = function(pixels, width, height, components) {
	io.Example.call(this, new Uint8ClampedArray(pixels));
	this.width = width;
	this.height = height;
	this.components = components || 1;
};

io.ImageExample.prototype = Object.create(io.Example.prototype);

io.ImageExample.prototype.typename = "io.ImageExample";

io.ImageExample.prototype.getPixel = function(x, y, component) {
	component = component || 0;
	return this.features[(y * this.width + x) * this.components + component];
}

io.ImageExample.prototype.setPixel = function(x, y, value) {
	component = component || 0;
	this.features[x + y * this.width] = value;
}


// Get an Image object for display in the browser.
// image: Optional Image object to be re-used (this will change the `src` attribute
//		of the image and thus completely replace any contents).
// canvas: Canvas element to be used. If not specified, a new <canvas> element is
//      allocated.
//
io.ImageExample.prototype.getImage = function(image, canvas) {
	canvas = canvas || document.createElement("canvas");
	if (canvas.width != this.width || canvas.height != this.height) {
		canvas.width = this.width;
		canvas.height = this.height;
	}
	var rgba_triples = new Uint8ClampedArray(this.width * this.height * 4);
	for (var src_cursor = 0, dest_cursor = 0; src_cursor < this.features.length; ++src_cursor) {
		rgba_triples[dest_cursor++] = this.features[src_cursor];
		rgba_triples[dest_cursor++] = this.features[src_cursor];
		rgba_triples[dest_cursor++] = this.features[src_cursor];
		rgba_triples[dest_cursor++] = 255;
	}
	var ctx = canvas.getContext('2d');
	ctx.putImageData(new ImageData(rgba_triples, this.width, this.height), 0, 0);

	image = image || new Image();
	image.src = canvas.toDataURL();
	return image;
};
